# HoliCity: A City-Scale Data Platform for Learning Holistic 3D Structures
<img src="https://people.eecs.berkeley.edu/~zyc/holicity/images/teaser.png">

This repository contains instructions and demo code of the paper:  [Yichao Zhou](https://yichaozhou.com/), [Jingwei Huang](http://haozhi.io/), [Xili Dai](https://github.com/Delay-Xili), [Linjie Luo](http://linjieluo.com/), [Zhili Chen](http://www.zhilichen.com/), [Yi Ma](https://people.eecs.berkeley.edu/~yima). ["HoliCity: A City-Scale Data Platform for Learning Holistic 3D Structures"](https://arxiv.org/abs/2008.03286). Technical Report. [arXiv:2008.03286](https://arxiv.org/abs/2008.03286) [cs.CV].

## Downloads

Please visit our [project website](https://holicity.io) for the overview and download links of the HoliCity dataset. We provide sample data in the folder [sample-data](https://github.com/zhou13/holicity/tree/master/sample-data).

## Panoramas

The panorama images are stored with [equirectangular projection](https://en.wikipedia.org/wiki/Equirectangular_projection).

## Images

We provide perspective renderings of panorama images. The field of views of all the current renderings are 90 degrees and the principal point is at the center of the image.  In other words, for a image with resolution <img src="https://latex.codecogs.com/svg.latex?512%20\times%20512">, the OpenCV camera instrinsic matrix is

<p align="center"><img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}256%20&%20&%20256%20\\%20&%20256%20&%20256%20\\&%20&%201\end{bmatrix}."/></p>

## Geolocations

<img src="https://people.eecs.berkeley.edu/~zyc/holicity/images/map_big.png">

HoliCity provides refined geolocation of each viewpoint and its corresponding perspective images in the coordinate of WGS-84, i.e., the longitude and latitude. The above figure shows the viewpoints on Google Maps. The following table summarizes the meanings of entries of geolocation annotations that are specific to **the viewpoints (panoramas)**.

| Entry        | Explanations                                                 |
| :----------- | ------------------------------------------------------------ |
| `loc`        | The xyz coordinate of the viewpoints in the space of the CAD models. We provide utility functions `model2gps` and `gps2model` in `gps_london.py` for converting between the xy coordinate of the CAD model and the WGS84 coordinate, i.e., longitude and latitude. The z coordinate represents the distance between the camera and the terrain. |
| `pano_yaw`   | Yaw of the panorama camera (panorama center) with respect to the north. |
| `tilt_yaw`   | Tilt direction of the panorama camera with respect to the north. Such tilt exists because the street-view car might be on a slope. |
| `tilt_pitch` | Tilt degree of the panorama camera.                          |

The following code snippet converts a point in the space of the CAD model to the space of a local viewpoint.

```python        loc, panoYaw, tiltYaw, tiltPitch = x[:3], x[3], x[4], x[5]
from vispy.util.transforms import rotate
def world_to_panorama(xyz, loc, pano_yaw, tilt_yaw, tilt_pitch):
    """Project xyz (a point in the space of the CAD model) to the space of a local viewpoint."""
    axis = np.cross([np.cos(pano_yaw), np.sin(tilt_yaw), 0], [0, 0, 1])
    R = (rotate(pano_yaw, [0, 0, 1]) @ rotate(tilt_pitch, axis))[:3, :3]
    return xyz @ R
```

The following code snippet draw the point in the space of a local viewpoint onto the corresponding panorama image.

```python
def draw_point_on_panorama(pp, panorama_image):
    """Draw pp (in the space of a local viewpoint) on the panorama image"""
    pp = pp / LA.norm(pp)
    pitch = math.atan(pp[2] / LA.norm(pp[:2]))
    yaw = math.atan2(pp[0], pp[1])
    x, y = (yaw + np.pi) / (np.pi * 2), (pitch + np.pi / 2) / np.pi
    plt.imshow(panorama_image)
    plt.scatter(x * img.shape[1], (1 - y) * img.shape[0])
    plt.show()
```

The following table summarizes the meanings of entries of geolocation annotations that are specific to **the perspective renderings**.

| Entry   | Explanations                                                 |
| ------- | ------------------------------------------------------------ |
| `R`     | The <img src="https://latex.codecogs.com/svg.latex?4\times4"> rotation and translation matrix that transforms the world coordinate of the CAD models to the camera space. This entry is derived from `loc`, `yaw`, and `pitch`. |
| `q`     | The rotational quaternion derived from `R`. Useful for training networks such as PoseNet. |
| `fov`   | The field of view.                                           |
| `yaw`   | The direction of the camera with respect to the north.       |
| `pitch` | The direction of the camera with respect to the horizontal plane. Example: 0 means cameras are pointed horizontally and 90 means cameras are pointed toward the sky. |
| `tilt`  | Currently, this entry is not used. All the perspective renderings have zero tilt, which means that the up-forward planes of cameras are always perpendicular to the horizontal plane. |

The following code snippet computes `R` from `loc`, `yaw`, and `pitch`.

````python
def transformation_matrix(loc, yaw, pitch):
    """Computes 4x4 world-to-camera transformation matrix"""
    yaw = -yaw * np.pi / 180 + np.pi / 2
    pitch = pitch * np.pi / 180
    return lookat(
        loc,
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)],
        [0, 0, 1],
    )

def lookat(position, forward, up=[0, 1, 0]):
    """Computes 4x4 transformation matrix to put camera looking at look point."""
    c = np.asarray(position).astype(float)
    w = -np.asarray(forward).astype(float)
    u = np.cross(up, w)
    v = np.cross(w, u)
    u /= LA.norm(u)
    v /= LA.norm(v)
    w /= LA.norm(w)
    return np.r_[u, u.dot(-c), v, v.dot(-c), w, w.dot(-c), 0, 0, 0, 1].reshape(4, 4)
````



## City CAD Models

Currently, [AccuCities](https://www.accucities.com/new-3d-london-samples-cover-full-square-kilometer/) provides freely available CAD models for an area of 1 km<sup>2</sup>  in the London city to the public. We label all the viewpoints of HoliCity in this area with suffix `_HD`. If you are using HoliCity for research purposes, you might want to contact with AccuCities and apply for other city models. The unit of the CAD models is the meter.

## Holistic Surface Segmentation

<img src="https://people.eecs.berkeley.edu/~zyc/holicity/images/surface-segmentations-pazo2.jpg"/>

We segment the surface of the 3D CAD model based on (approximate) local curvature.  The **reference MaskRCNN implementation** used in our paper can be found [here (HoliCity-MaskRCNN)](https://github.com/Delay-Xili/HoliCity-MaskRCNN).  You should be able to use it to reproduce the results of our paper.

### 3D Planes

For each surface segment, we approximate it with a 3D plane whose equation is <img src="https://latex.codecogs.com/gif.latex?%5Cinline%20w%5ETx&amp;plus;1%3D0">. We provide the parameter <img src="https://latex.codecogs.com/gif.latex?%5Cinline%20w"> for the fitted plane of each surface segment.  The plane fitting is done on the global level of the CAD model.  Surface segments and planes exclude trees.

`plane.py`  provides the example code showing how to parse the plane parameters and draw depth maps and normal maps accordingly. We note that there is some difference between the ground truth depth maps and the depth maps derived from the parameter <img src="https://latex.codecogs.com/gif.latex?%5Cinline%20w"> due to the error from global plane fitting, especially for large planes such as the ground.

## Coordinate Systems
For normal maps and vanishing points, the coordinate system of the camera in HoliCity follows the convention of OpenGL: The camera is placed at (0, 0, 0).  The x axis is toward the right of the image and the y axis is upward. The z axis points out of the screen to form the right-hand coordinate system, which means that the image plane is at  <img src="https://latex.codecogs.com/svg.latex?z=-1">.

## Low-level 3D Representations

We provide renderings of depth maps and normal maps for each perspective image. The unit of depth maps is the meter, which is the same as the unit of the CAD model. Renderings with the suffix `_HD` have more details than the renderings with the `_LD` suffix. *We note that the low-level represesntaions currently do not include moving objects such as cars and pedestrians.*

## Vanishing Points

We provide the extracted vanishing points using the script `normal2vpts.py`. 

## Semantic Segmentation

We provide the semantic segmentation for all the perspective images. The following table shows the meaning of the labels.

| Values | Meaning        |
| ------ | -------------- |
| 0      | Sky or nothing |
| 1      | Buildings      |
| 2      | Roads          |
| 3      | Terrains       |
| 4      | Trees          |
| 5      | Others         |

## Acknowledgment

This work is sponsored by a generous grant from Sony Research US. We'd also like to thank Sandor Petroczi and Michal Konicek from AccuCities for the help of their London CAD model.

## Reference

If you find HoliCity useful in your research, please consider citing:          

```
@article{zhou2020holicity,
    author={Zhou, Yichao and Huang, Jingwei and Dai, Xili and Luo, Linjie and Chen, Zhili and Ma, Yi},
    title={{HoliCity}: A City-Scale Data Platform for Learning Holistic {3D} Structures},
    year = {2020},
    archivePrefix = "arXiv", 
    note = {arXiv:2008.03286 [cs.CV]},
}
```
