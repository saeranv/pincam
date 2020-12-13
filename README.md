## pincam-simple
A simple pinhole camera library developed in Numpy, used mainly for educational purposes. 

It computes a camera projection matrix to transform 3D geometries to 2D geometries, and incorporates a simple raycasting method to resolve depth order for overlapping geometries. Visualization is then achieved using geopandas.

The "simple" in pincam-simple refers to the implementation of the raycasting method, which is done manually using Numpy and for loops and is thus extremely slow. This is why this repo's main purpose is to explore how camera projection works.

### Example
Snapshot of three surfaces (with some complicated overlapping):
![x](/resources/imgs/box_example.PNG "x")

Using pincam to illustrate the transformation (or squishing) of 3D geometries in the camera view frustum:
![x](/resources/imgs/view_frustrum.png "x")
