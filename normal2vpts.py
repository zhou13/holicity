import argparse
import math
import os.path as osp
import sys
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist, euclidean
from sklearn.cluster import DBSCAN

DBSCAN_EPS = 5
DBSCAN_MIN_SAMPLES = 60
MIN_SAMPLES = 125
FOCAL_LENGTH = 1


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def handle(prefix):
    print(prefix)

    pitch = math.radians(float(prefix[-2:]))
    gnd = np.array([0, math.cos(pitch), -math.sin(pitch)])
    # print(gnd)

    fnrml = f"{prefix}_nrml.npz"

    nrml = np.load(fnrml)["normal"].copy()
    nrml = cv2.resize(nrml, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)

    nrml = nrml.reshape([-1, 3])
    nrml_nz = nrml
    dist = np.arccos(np.clip(np.abs(nrml_nz @ nrml_nz.T), 0, 1)) / np.pi * 180
    clusters = DBSCAN(
        eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed"
    ).fit(dist)
    label = clusters.labels_.copy()

    unique, counts = np.unique(label, return_counts=True)

    vpts = [(gnd, 1000)]
    for idx, cnt in zip(unique, counts):
        if idx == -1:
            continue
        n = geometric_median(nrml[label == idx, :])
        n /= LA.norm(n)
        if math.acos(np.clip(abs(n @ gnd), 0, 1)) < math.radians(10):
            continue
        n = np.cross(n, gnd)
        n /= LA.norm(n)
        # print("vpts:", n, idx, cnt)
        vpts += [(n, cnt)]
    vpts.sort(key=lambda x: -x[1])
    vpts, confidence = zip(*vpts)
    np.savez(f"{prefix}_vpts.npz", vpts=vpts, confidence=confidence)

    # print(vpts)
    # print(confidence)

    # fimag = f"{prefix}_imag.png"
    # fmesh = f"{prefix}_mesh.png"
    # I = cv2.imread(fimag)
    # Imesh = cv2.imread(fmesh)
    # fimag = f"{prefix}_imag.png"
    # I = cv2.imread(fimag)
    # plt.imshow(I)
    # cc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # for c, w, conf in zip(cc, vpts, confidence):
    #     if conf < MIN_SAMPLES:
    #         break
    #     x = w[0] / -w[2] * FOCAL_LENGTH * 256 + 256
    #     y = -w[1] / -w[2] * FOCAL_LENGTH * 256 + 256
    #     plt.scatter(x, y, color=c)
    #     for xy in np.linspace(0, 512, 10):
    #         plt.plot(
    #             [x, xy, x, xy, x, 0, x, 511], [y, 0, y, 511, y, xy, y, xy], color=c,
    #         )
    #     plt.xlim(0, 511)
    #     plt.ylim(511, 0)
    # plt.show()
    # return

    # plt.figure()
    # plt.imshow(nrml.reshape([64, 64, 3]) / 2 + 0.5)
    # plt.show()


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", help="path to the index of pano images")
    parser.add_argument("--batch", type=int, default=0, help="parallel processing")
    parser.add_argument("--total", type=int, default=1, help="parallel procesing")
    # fmt: on
    args = parser.parse_args()

    flist = glob(args.glob)
    flist.sort()
    for f in flist[args.batch :: args.total]:
        handle(f[:-9])


if __name__ == "__main__":
    main()
