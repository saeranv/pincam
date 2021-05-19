## Pincam
A simple pinhole camera library in Python.

Pincam computes a camera projection matrix that projects 3D geometries onto a defined camera image plane. It incorporates a simple raycasting method to resolve depth order for geometries relative to the image plane. The geometries are stored in a geopandas DataFrame, which facilitates easy plotting and customization of the geometries for further visualization and analysis.

For example, here's a simple analysis I did at work with Pincam and geopandas to calculate surface solar insolation:

![x](/resources/imgs/solar_analysis.png "x")

### Examples
An example project showing the visualization of three surfaces with a heading and pitch of 15 degrees, and focal length of 25 mm:
![x](/resources/imgs/box_example_2.PNG "x")

Using Pincam to show how camera projection works, here we see how 3D geometries are scaled by depth within the camera view frustum:
![x](/resources/imgs/view_frustrum.png "x")

These examples can be seen in the quickstart.ipynb and view_frustrum.ipynb notebooks, in the notebooks directory.


### TODO:

1. Separate out Pincam class into Raycast, Render and Pincam classes.
2. Get rid of OpenCV dependency.
3. Get rid of Ladybug-geometry dependency and replace with meshing algorithm.
1. Rewrite raycasting library to use parallelized monte-carlo sampling.