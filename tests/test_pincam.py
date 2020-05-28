import pytest
import os
import json
from pprint import pprint as pp

import numpy as np
from pincam import pincam as cam
from pincam.matrix_utils2 import MatrixUtils2 as mu


def r(d):
    return d / 180.0 * np.pi


def load_fixture_array(fpath):
    """Load array of fixture"""
    with open(fpath, 'r') as f:
        fixdata = json.loads(f.read())

    return np.array(fixdata)


def write_fixture_array(fpath, arr):
    """Write array to fixture json"""

    lsts = arr.tolist()
    with open(fpath, 'w') as f:
        f.write(json.dumps(lsts))


def test_basic_transform():
    """To grok order"""
    mtx1 = np.array([
        [1,  0,  0,  0],
        [0,  0,  3,  0],
        [0,  2,  0,  0],
        [0,  0,  0,  1]
    ])

    mtx2 = np.array([
        [1,  0,  0,  0],
        [0,  1,  0,  5],
        [0,  0,  1,  10],
        [0,  0,  0,  1]
    ])

    refmtx = np.array([
        [1,  0,  0,  0],
        [0,  0,  3,  5],
        [0,  2,  0,  10],
        [0,  0,  0,  1]
    ])

    # Current attempt with points, doesn't work on matrices
    out1 = np.matmul(mtx1, mtx2.T).T
    assert not np.allclose(out1, refmtx)

    out2 = mtx1.dot(mtx2) # Won't work b/c matrix multiplication is from left to right
    assert not np.allclose(out2, refmtx, 1e-10)

    out3 = mtx2.dot(mtx1) # Correct order
    assert np.allclose(out3, refmtx)

    # Same order with matmul is equivalent
    out4 = np.matmul(mtx2, mtx1)
    assert np.allclose(out4, refmtx)


def test_rotation_transform():
    """ Test rotation matrix"""
    # Rotation around x axis by 90
    ref_rmtxX = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1]
    ])

    origin = np.array([0, 0, 0])
    axis = np.array([1, 0, 0])
    theta = r(90.0)
    chkmtx = mu.xform_rotation_matrix(origin, axis, theta)
    assert np.allclose(chkmtx, ref_rmtxX, 1e-10)

    # Rotation around z acis by 180
    ref_rmtxZ = np.array([
        [-1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ])

    axis = np.array([0, 0, 1])
    theta = r(180)
    chkmtx = mu.xform_rotation_matrix(origin, axis, theta)
    assert np.allclose(chkmtx, ref_rmtxZ, 1e-10)


def test_world_to_camera_transform():
    """Test world to camera matrix"""

    # Collection of points that we are going to flip to camera pose
    points = np.array([
        [0,  0,  4,  1],
        [2,  0,  0,  1],
        [0,  3,  0,  1],
        [1,  1,  0,  1]
    ])

    # Invert x, swap y, z
    chk_points = np.array([
        [ 0,  4,  0,  1],
        [-2,  0,  0,  1],
        [ 0,  0,  3,  1],
        [-1,  0,  1,  1]
    ])

    # Test
    cammtx = cam.world_to_camera_matrix()
    # For points, remember to transpose to points into columns
    xpts = np.matmul(cammtx, points.T).T

    assert np.allclose(xpts, chk_points, 1e-10)


def test_camera_to_world_transform():
    """Test camera to world matrix"""

    # Collection of points that we are going to flip to camera pose
    world_points = np.array([
        [0,  0,  4,  1],
        [2,  0,  0,  1],
        [0,  3,  0,  1],
        [1,  1,  0,  1]
    ])

    # Invert x, swap y, z
    cam_points = np.array([
        [ 0,  4,  0,  1],
        [-2,  0,  0,  1],
        [ 0,  0,  3,  1],
        [-1,  0,  1,  1]
    ])

    # Test
    worldmtx = cam.camera_to_world_matrix()
    # For points, remember to transpose to points into columns
    xpts = np.matmul(worldmtx, cam_points.T).T

    assert np.allclose(xpts, world_points, 1e-10)


def test_post_order_transform_multiplication():
    """Test post-order matrix multiplication of rigid xforms"""

    # Make the reference matrices

    # Translation 0, 5, 10
    ref_tmtx = np.array([
        [1,  0,  0,  0],
        [0,  1,  0,  5],
        [0,  0,  1,  10],
        [0,  0,  0,  1]
    ])

    # Rotation around x axis by 90
    ref_rmtxX = np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1]
    ])

    # Rotation around z acis by 180
    ref_rmtxZ = np.array([
        [-1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ])

    # Output matrix
    ref_emtx = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  5],
        [ 0, -1,  0, 10],
        [ 0,  0,  0,  1]
    ])


    # Test combination of xforms
    # Order for rigid xforms, by convention:
    # 1. Rotation z
    # 2. Rotation x
    # 3. Translation
    # E = T(X(Z))

    emtx = np.matmul(ref_rmtxX, ref_rmtxZ)
    emtx = np.matmul(ref_tmtx, emtx)

    assert np.allclose(emtx, ref_emtx, 1e-10)


def test_extrinsic_matrix():
    """Extrinsic matrix"""

    # Rotation ccw around x axis by 45 degrees
    # up by 5 units
    heading = 0.0
    pitch = np.pi / 4.
    cam_posn = np.array([0, 0, 5])
    Rt = cam.extrinsic_matrix(heading, pitch, cam_posn)

    v = 0.7071067811865476 # sin(45) and cos(45)
    chk_Rt = np.array([
        [ 1,  0,  0,  0],
        [ 0,  v, -v,  0],
        [ 0,  v,  v, -5],
        [ 0,  0,  0,  1]])

    assert np.allclose(Rt, chk_Rt, atol=1e-5)

    # Test world to camera transform
    wc = cam.world_to_camera_matrix()
    Rtc = np.matmul(wc, Rt)
    chk_Rtc = np.array([
        [ -1,  0,  0,  0],
        [  0,  v,  v, -5],
        [  0,  v, -v,  0],
        [  0, 0, 0, 1]])

    assert np.allclose(Rtc, chk_Rtc, atol=1e-5)


def test_intrinsic_matrix():
    """Intrinsic matrix"""

    # Hardcode parameters
    fov = r(45)
    sensor_width = 2 * np.tan(fov / 2.0) * 0.1
    S = 0.1 * (100. / sensor_width) / 100.
    cx, cy = 0, 0

    K = np.array([
        [S,  0,  cx,  0],
        [0,  S,  cy,  0],
        [0,  0,  1,   0]])

    chk_K = cam.intrinsic_matrix(fov)

    assert np.allclose(K, chk_K, atol=1e-5)


def test_perspective_projection_transform():
    """Test perspective transformation"""
    v = 0.7071
    w = 1.4142
    m = w + v
    Rtc = np.array([
        [-1,  0,  0,  0 ],
        [ 0,  v,  v, -5 ],
        [ 0,  v, -v, -10],
        [ 0,  0,  0,  1 ]
    ])
    K = np.array([
        [2,  0,  1,  0],
        [0,  2,  1,  0],
        [0,  0,  1,  0]
    ])

    # Check projection matrix
    P = np.matmul(K, Rtc)

    chk_P = np.array([
        [-2,  v,  -v, -10 ],
        [ 0,  m,   v, -20 ],
        [ 0,  v,  -v, -10]
    ])

    assert np.allclose(P, chk_P, atol=1e-5)

    # Check projection of segment left of the origin
    seg_left = np.array([
        [-5, 0, 0, 1],
        [-5, 5, 0, 1]
    ])

    chk_prj = np.array([
        [ 0, -20, -10],
        [ 3.5355, -9.3935, -6.4645]
    ])
    prj = np.matmul(P, seg_left.T).T

    assert np.allclose(prj, chk_prj, atol=1e-2)


def test_zbuffer():
    """Testing ordering of surfaces from zbuffering"""

    fov = r(45)
    heading = 0
    pitch = 45
    cam_posn = np.array([0, -5, 10])
    P = cam.projection_matrix(fov, heading, pitch, cam_posn)

    # Test ordering of points in center
    seg_center = np.array([
        [0, 0, 0, 1], # pt_front
        [0, 5, 0, 1]  # pt_back
    ])

    prj_center = np.matmul(P, seg_center.T).T
    w = prj_center[:, 2]
    pt_front = prj_center[0] / w[0]
    pt_back = prj_center[1] / w[1]

    # pp(pt_front)
    # pp(pt_back)

    # z coordinate of front point should be less then back
    assert pt_front[1] < pt_back[1]
    # x coordinates should be equal
    assert np.abs(pt_front[0] - pt_back[0]) < 1e-10
    # confirm x coincident w/ principle point
    assert np.abs(pt_front[0]) < 1e-10 and np.abs(pt_back[0]) < 1e-10



    # Test ordering of points to left
    seg_left = np.array([
        [-5, 0, 0, 1], # pt_front
        [-5, 5, 0, 1]  # pt_back
    ])

    prj_left = np.matmul(P, seg_left.T).T
    w = prj_center[:, 2]
    pt_front = prj_left[0] / w[0]
    pt_back = prj_left[1] / w[1]

    # pp(pt_front)
    # pp(pt_back)

    # z coordinate of front point should be less then back
    assert pt_front[1] < pt_back[1]
    # x coordinate of front should be < back
    assert pt_front[0] < pt_back[0]


def test_bounding_box():
    # Calculate bbox from matrix of points
    ptmtx = np.array([
        [-10, 10,   0],
        [10,   5,  20],
        [10,   5,  20],
        [3,   12, -20]
    ])

    chk_bbox = np.array([
        [[-10, 12, -20], [10, 12, -20],
         [10, 5, -20], [-10, 5, -20]],
        [[-10, 5, 20], [10, 5, 20],
         [10, 12,  20], [-10, 12,  20]]
    ])

    bbox = cam._bounding_box(ptmtx)

    assert np.allclose(bbox, chk_bbox, atol=1e-10)


def test_single_plane_bounding_box():
    # Test bounding box when you get one surface
    ptmtx = np.array(
        [[-4, 0, 0], [4, 0, 0], [4, 0, 6], [0, 0, 10], [-4, 0, 6]]
    )

    bot, top = cam._bounding_box(ptmtx)

    # Single line
    # np.array(
    #     [[-4,  0,  0],
    #      [ 4,  0,  0],
    #      [ 4,  0,  0],
    #      [-4, 0, 0]]
    # )

    assert False

def test_surface_normal():
    # Test surface normal from ccw surface
    srf1 = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0]
    ])

    srf2 = srf1[::-1]

    assert np.allclose(
        [0, 0, 1], cam._surface_normal(srf1))

    assert np.allclose(
        [0, 0, -1], cam._surface_normal(srf2))


def test_simple_view_factor():
    # Test view of simple vertical plane with no rotation
    srf = np.array([
        [-5, 0, 0], [5, 0, 0], [5, 0, 10], [-5, 0, 10]
    ])
    r = lambda d: d / 180. * np.pi
    fov = r(35)
    heading = 0
    pitch = 0
    cam_posn = np.array([0, -10, 5])
    P, Rtc = cam.projection_matrix(fov, 2, heading, pitch, cam_posn.copy())

    view_factor = cam.view_factor(Rtc, srf) > 0.0
    assert view_factor == True

    view_factor = cam.view_factor(Rtc, srf[::-1]) > 0.0
    assert view_factor == False

    # Test view of simple horizontal plane with no rotation
    srf = np.array([
        [-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]
    ])
    r = lambda d: d / 180. * np.pi
    fov = r(35)
    heading = 0
    pitch = 0
    cam_posn = np.array([0, -10, 5])
    P, Rtc = cam.projection_matrix(fov, 2, heading, pitch, cam_posn.copy())
    # Should be exactly perpendicular
    view_factor = cam.view_factor(Rtc, srf) > 0.0
    assert view_factor == True

    view_factor = cam.view_factor(Rtc, srf[::-1]) > 0.0
    assert view_factor == False


def test_complex_view_factor():
    # Check view_hemisphere for each geoms
    bot_srf = np.array([
        [-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]
    ])

    top_srf = np.array([
        [-5, -5, 9], [5, -5, 9], [5, 5, 9], [-5, 5, 9]
    ])

    # P matrix see inner face of bottom and top faces
    r = lambda d: d / 180. * np.pi
    fov = r(35)
    heading = r(45)
    pitch = r(-10)
    cam_posn = np.array([0, -15, 7])
    P, Rtc = cam.projection_matrix(fov, 2, heading, pitch, cam_posn.copy())

    view_factor = cam.view_factor(P, bot_srf) > 0.0
    assert view_factor == True

    view_factor = cam.view_factor(P, bot_srf[::-1]) > 0.0
    assert view_factor == False

    # Look at top surface
    view_factor = cam.view_factor(P, top_srf) > 0.0
    assert view_factor == False

    view_factor = cam.view_factor(P, top_srf[::-1]) > 0.0
    assert view_factor == True

    # Look at underside of bottom surface
    pitch = r(-55)
    P, Rtc = cam.projection_matrix(fov, 2, heading, pitch, cam_posn.copy())

    view_factor = cam.view_factor(P, bot_srf) > 0.0
    assert view_factor == False

    view_factor = cam.view_factor(P, bot_srf[::-1]) > 0.0
    assert view_factor == True

    # P matrix see inner face of bottom and top faces
    pitch = r(35)
    P, Rtc = cam.projection_matrix(fov, 2, heading, pitch, cam_posn.copy())

    view_factor = cam.view_factor(P, bot_srf) > 0.0
    assert view_factor == True

    view_factor = cam.view_factor(P, bot_srf[::-1]) > 0.0
    assert view_factor == False

    view_factor = cam.view_factor(P, top_srf) > 0.0
    assert view_factor == True

    view_factor = cam.view_factor(P, top_srf[::-1]) > 0.0
    assert view_factor == False


def test_simple_snapshot():
    # Test projection of three simple surfaces

    # Define surfaces
    bot_srf = np.array(
        [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]]
        )
    top_srf = np.array(
        [[-5, -5, 10], [5, -5, 10], [5, 5, 10], [-5, 5, 10]]
        )
    vrt_srf = np.array(
        [[-4, 0, 0], [4, 0, 0], [4, 0, 6], [0, 0, 10], [-4, 0, 6]]
        )
    geoms = [vrt_srf, top_srf, bot_srf]

    # Set camera parameters
    fov = r(35)
    heading = r(15)
    pitch = r(25)
    cam_posn = np.array([0, -35, 4])
    P, Rtc = cam.projection_matrix(fov, 5, heading, pitch, cam_posn)
    xgeoms = cam.project_by_z(P, P, cam_posn, geoms, False)

    # Define the xgeoms we should get, in correct order
    fpath = os.path.join('tests', 'fixtures', 'simple_snapshot_surfaces.json')
    chk_xgeoms = load_fixture_array(fpath)

    # Assert
    for xgeom, chk_xgeom in zip(xgeoms, chk_xgeoms):
        assert np.allclose(xgeom, chk_xgeom, atol=1e-10)


if __name__ == "__main__":
    test_basic_transform()
    test_rotation_transform()
    test_world_to_camera_transform()
    test_camera_to_world_transform()
    test_post_order_transform_multiplication()
    test_extrinsic_matrix()
    #test_intrinsic_matrix()
    #test_perspective_projection_transform()
    #test_zbuffer()
    test_bounding_box()
    #test_single_plane_bounding_box()
    test_surface_normal()
    test_simple_view_factor()
    test_complex_view_factor()
    test_simple_snapshot()

