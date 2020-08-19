# HoliCity: A City-Scale Data Platform for Learning Holistic 3D Structures
<img src="https://people.eecs.berkeley.edu/~zyc/holicity/images/teaser.png">

This repository contains instructions and demo code of the paper:  [Yichao Zhou](https://yichaozhou.com/), [Jingwei Huang](http://haozhi.io/), [Xili Dai](https://github.com/Delay-Xili), [Linjie Luo](http://linjieluo.com/), [Zhili Chen](http://www.zhilichen.com/), [Yi Ma](https://people.eecs.berkeley.edu/~yima). ["HoliCity: A City-Scale Data Platform for Learning Holistic 3D Structures"](https://arxiv.org/abs/2008.03286). Technical Report. [arXiv:2008.03286](https://arxiv.org/abs/2008.03286) [cs.CV].

## Downloads

Please visit our [project website](https://holicity.io) for the overview and download links of the HoliCity dataset. We provide sample data in the folder [sample-data](https://github.com/zhou13/holicity/tree/master/sample-data).

## Panoramas

The panorama images are stored with [equirectangular projection](https://en.wikipedia.org/wiki/Equirectangular_projection).

## Images

We provide perspective renderings of panorama images. The field of views of the current renderings are 90 degrees.

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
| `fov`   | Field of views of panorama renderings.                       |
| `yaw`   | The direction of the camera with respect to the north.       |
| `pitch` | The direction of the camera with respect to the horizontal plane. Example: 0 means cameras are pointed horizontally and 90 means cameras are pointed toward the sky. |
| `tilt`  | Currently, this entry does not exist. All the perspective renderings have zero tilt, which means that the up-forward planes of cameras are always perpendicular to the horizontal plane. |

## City CAD Models

Currently, [AccuCities](https://www.accucities.com/new-3d-london-samples-cover-full-square-kilometer/) provides freely available CAD models for an area of 1 km<sup>2</sup>  in the London city to the public. We label all the viewpoints of HoliCity in this area with suffix `_HD`. If you are using HoliCity for research purposes, you might want to contact with AccuCities and apply for other city models. The unit of the CAD models is the meter.

## Holistic Surface Segmentation

We segment the surface of the 3D CAD model based on the (approximate) local curvature.  The **reference MaskRCNN implementation** used in our paper can be found [here](https://github.com/Delay-Xili/HoliCity-MaskRCNN).

<img src="https://people.eecs.berkeley.edu/~zyc/holicity/images/surface-segmentations-pazo2.jpg">

### 3D Planes

For each surface segment, we approximate it with a 3D plane whose equation is <img src="https://latex.codecogs.com/gif.latex?%5Cinline%20w%5ETx&amp;plus;1%3D0">. We provide the parameter <img src="https://latex.codecogs.com/gif.latex?%5Cinline%20w"> for the fitted plane of each surface segment.  The plane fitting is done on the global level of the CAD model.  Surface segments and planes exclude trees.

`plane.py`  provides the example code showing how to parse the plane parameters and draw depth maps and normal maps accordingly. We note that there is some difference between the ground truth depth maps and the depth maps derived from the parameter <img src="https://latex.codecogs.com/gif.latex?%5Cinline%20w"> due to the error from global plane fitting, especially for large planes such as the ground.

## Low-level 3D Representations

We provide renderings of depth maps and normal maps for each panorama image. The unit of depth maps is the meter. We note that HoliCity does not include moving objects such as cars and pedestrians for now.

## Semantic Segmentation

We provide the semantic segmentation for perspective images. The following table shows the meaning of the labels.

| Values | Meaning        |
| ------ | -------------- |
| 0      | Sky or nothing |
| 1      | Buildings      |
| 2      | Roads          |
| 3      | Terrains       |
| 4      | Trees          |
| 5      | Others         |



## Vanishing Points

We provide the extracted vanishing points using the script `normal2vpts.py`.

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
