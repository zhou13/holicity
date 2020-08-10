import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

# palette = sns.color_palette("tab20")
palette = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (1.0, 0.7333333333333333, 0.47058823529411764),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (1.0, 0.596078431372549, 0.5882352941176471),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
]


def main():
    prefix = "sample-data/GWUmH4qmANxNTVk_f-_Wrw_HD_060_20"

    I = cv2.imread(f"{prefix}_imag.png")[:, :, ::-1] / 255.0
    buf_plane = cv2.imread(f"{prefix}_plan.png", -1)
    with np.load(f"{prefix}_plan.npz") as f:
        plane_normal = f["ws"]

    # load ground truth dense map
    with np.load(f"{prefix}_nrml.npz") as f:
        buf_nrml_gt = f["normal"]
    with np.load(f"{prefix}_dpth.npz") as f:
        buf_dpth_gt = f["depth"][..., 0]

    # draw a nice overlay
    for i in range(len(plane_normal)):
        alpha_fill = (buf_plane == i + 1)[..., None].astype(np.float32)
        sx = cv2.Sobel(alpha_fill, cv2.CV_32F, 1, 0, ksize=5)
        sy = cv2.Sobel(alpha_fill, cv2.CV_32F, 0, 1, ksize=5)
        alpha_edge = (sx ** 2 + sy ** 2) ** 0.5
        alpha_edge /= max(0.001, np.max(alpha_edge))
        alpha_edge = alpha_edge[..., None]
        alpha_fill *= 0.5
        color = palette[random.randrange(len(palette))]
        I = I * (1 - alpha_fill) + alpha_fill * color
        I = I * (1 - alpha_edge) + alpha_edge * color

    plt.figure(), plt.title("overlay"), plt.imshow(I)
    plt.figure(), plt.title("nrml_gt"), plt.imshow(buf_nrml_gt / 2 + 0.5)
    plt.figure(), plt.title("dpth_gt"), plt.imshow(buf_dpth_gt)

    # Camera coordinate is with OpenGL convention
    buf_nrml_p = np.zeros((512, 512, 3,))
    buf_dpth_p = np.zeros((512, 512), dtype=float)
    x, y = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(1, -1, 512))
    xyz = np.stack([x, y, -np.ones_like(x)], axis=2)
    for p, w in enumerate(plane_normal):
        ps = buf_plane == p + 1
        buf_nrml_p[ps] = -w / LA.norm(w)
        buf_dpth_p[ps] = 1 / (xyz[ps] @ w)
    buf_dpth_p[buf_dpth_p < 0] = 0
    buf_dpth_p[buf_dpth_p > 1000] = 1000
    plt.figure(), plt.title("nrml_from_plane"), plt.imshow(buf_nrml_p / 2 + 0.5)
    plt.figure(), plt.title("dpth_from_plane"), plt.imshow(buf_dpth_p)

    plt.show()


if __name__ == "__main__":
    main()
