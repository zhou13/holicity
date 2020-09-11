import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

from vispy import app, gloo
from vispy.gloo import gl
from vispy.gloo.wrappers import read_pixels
from vispy.util.ptime import time
from vispy.util.transforms import perspective, rotate, translate

PANO_VERT = """
uniform mat4 u_view;
uniform mat4 u_model;
uniform mat4 u_projection;

attribute vec3 a_position;

varying   vec3 v_position;

void main()
{
    v_position = a_position;
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
"""

PANO_FRAG = """
const float PI   = 3.141592653589793;
const float PI2  = 6.283185307179586;
const float PI_2 = 1.570796326794896;

varying vec3 v_position;

// uniform float alpha;
uniform sampler2D image;

void main()
{
    vec3 v = normalize(v_position);
    float pitch = atan(v.z / sqrt(v.x * v.x + v.y * v.y));
    float yaw = atan(v.x, v.y);
    vec2 uv = vec2((yaw + PI) / PI2, (pitch + PI_2) / PI);
    vec4 pixel = vec4(uv.x, uv.x, uv.x, 1);
    pixel = texture2D(image, uv);
    gl_FragColor = pixel;
}
"""

CUBE_V = [
    [-1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [-1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [1.0, -1.0, 1.0],
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, -1.0, -1.0],
    [-1.0, -1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [1.0, -1.0, 1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, -1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0],
]


def lookat(position, forward, up=[0, 1, 0]):
    """Computes matrix to put camera looking at look point."""
    c = np.asarray(position).astype(float)
    w = -np.asarray(forward).astype(float)
    u = np.cross(up, w)
    v = np.cross(w, u)
    u /= LA.norm(u)
    v /= LA.norm(v)
    w /= LA.norm(w)
    return np.r_[u, u.dot(-c), v, v.dot(-c), w, w.dot(-c), 0, 0, 0, 1].reshape(4, 4).T


def main():
    # Load panorama and perspective camera parameters
    with np.load("sample-data/GWUmH4qmANxNTVk_f-_Wrw_HD_060_20_camr.npz") as f:
        # Load panorama parameters
        pano_yaw = f["pano_yaw"]
        tilt_yaw = f["tilt_yaw"]
        tilt_pitch = f["tilt_pitch"]
        # Load perspective camera parameters
        fovy = f["fov"]
        yaw = f["yaw"]
        pitch = f["pitch"]
    image = cv2.imread("sample-data/GWUmH4qmANxNTVk_f-_Wrw.jpg")

    prog_pano = gloo.Program(PANO_VERT, PANO_FRAG)
    prog_pano["u_model"] = np.eye(4, dtype=np.float32)
    prog_pano["a_position"] = CUBE_V

    yaw = -yaw * np.pi / 180 + np.pi / 2
    pitch = pitch * np.pi / 180
    prog_pano["u_view"] = (
        rotate(pano_yaw, [0, 0, 1])
        @ rotate(
            tilt_pitch, np.cross([np.cos(tilt_yaw), np.sin(tilt_yaw), 0], [0, 0, 1]),
        )
        @ lookat(
            [0, 0, 0],
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)],
            [0, 0, 1],
        )
    )
    prog_pano["u_projection"] = perspective(fovy, 1.0, 0.01, 1e3)
    prog_pano["image"] = gloo.Texture2D(image[::-1, :, ::-1], interpolation="linear")

    gloo.clear((0, 0, 0, 0))
    gloo.set_viewport(0, 0, 512, 512)
    prog_pano.draw("triangles")
    buf_pano = read_pixels((0, 0, 512, 512), out_type=np.float32)[..., :3]

    # this resulting image should match sample-data/GWUmH4qmANxNTVk_f-_Wrw.jpg
    plt.imshow(buf_pano)
    plt.show()


if __name__ == "__main__":

    class Canvas(app.Canvas):
        def __init__(self):
            app.Canvas.__init__(self, show=False)
            self.update()

        def on_draw(self, event):
            main()
            app.quit()

    app.use_app("glfw")
    c = Canvas()
    app.run()
