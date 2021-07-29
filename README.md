## Pincam
A simple pinhole camera library in Python.

Pincam computes a camera projection matrix that projects 3D geometries onto a defined camera image plane. It incorporates a simple raycasting method to resolve depth order for geometries relative to the image plane. The geometries are stored in a geopandas DataFrame, which facilitates easy plotting and customization of the geometries for further visualization and analysis.

For example, here's a radiation analysis visualized with Pincam for one of my projects ([DeepRad](https://github.com/saeranv/DeepRad), a deep learning framework for building radiation prediction). The initial rows illustrates input geometries, and the final row illustrates the surface simulation results.

![x](/resources/imgs/in_out.PNG "x")
![x](/resources/imgs/in_out2.PNG "x")

### Examples
An example project showing the visualization of three surfaces with a heading and pitch of 15 degrees, and focal length of 25 mm:
![x](/resources/imgs/box_example_2.PNG "x")

Using Pincam to show how camera projection works, here we see how 3D geometries are scaled by depth within the camera view frustum:
![x](/resources/imgs/view_frustrum.png "x")

These examples can be seen in the quickstart.ipynb and view_frustrum.ipynb notebooks, in the notebooks directory.


### TODO:

1. Separate out Pincam class into matrix module, and Pincam class.
2. Seperate out Raycast, and Render modules.
X3. Get rid of OpenCV dependency.
4. Get rid of Ladybug-geometry dependency and replace with meshing algorithm.
5. Rewrite raycasting library to use parallelized monte-carlo sampling.


### RELEASES

1. Python package
2. Notebook of examples