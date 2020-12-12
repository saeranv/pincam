import pytest
import os
import json
from pprint import pprint as pp

import numpy as np
from pincam import Pincam
from pincam.matrix_utils2 import MatrixUtils2 as mu
cam = Pincam


def r(d):
    return d / 180.0 * np.pi


def load_fixture_array(fpath):
    """Load array of fixture"""
    with open(fpath, 'r') as f:
        fixdata = json.loads(f.read())

    return fixdata


def write_fixture_array(fpath, arr):
    """Write array to fixture json"""

    lsts = arr.tolist()
    with open(fpath, 'w') as f:
        f.write(json.dumps(lsts))


def test_basic_transform():
    """To grok order"""
    mtx1 = np.array([
        [1, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 1]
    ])

    mtx2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 5],
        [0, 0, 1, 10],
        [0, 0, 0, 1]
    ])

    refmtx = np.array([
        [1, 0, 0, 0],
        [0, 0, 3, 5],
        [0, 2, 0, 10],
        [0, 0, 0, 1]
    ])

    # Current attempt with points, doesn't work on matrices
    out1 = np.matmul(mtx1, mtx2.T).T
    assert not np.allclose(out1, refmtx)

    # Won't work b/c matrix multiplication is from left to right
    out2 = mtx1.dot(mtx2)
    assert not np.allclose(out2, refmtx, 1e-10)

    out3 = mtx2.dot(mtx1)  # Correct order
    assert np.allclose(out3, refmtx)

    # Same order with matmul is equivalent
    out4 = np.matmul(mtx2, mtx1)
    assert np.allclose(out4, refmtx)


def test_rotation_transform():
    """ Test rotation matrix"""
    # Rotation around x axis by 90
    ref_rmtxX = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    origin = np.array([0, 0, 0])
    axis = np.array([1, 0, 0])
    theta = r(90.0)
    chkmtx = mu.xform_rotation_matrix(origin, axis, theta)
    assert np.allclose(chkmtx, ref_rmtxX, 1e-10)

    # Rotation around z acis by 180
    ref_rmtxZ = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    axis = np.array([0, 0, 1])
    theta = r(180)
    chkmtx = mu.xform_rotation_matrix(origin, axis, theta)
    assert np.allclose(chkmtx, ref_rmtxZ, 1e-10)


def test_world_to_camera_transform():
    """Test world to camera matrix"""

    # Collection of points that we are going to flip to camera pose
    points = np.array([
        [0, 0, 4, 1],
        [2, 0, 0, 1],
        [0, 3, 0, 1],
        [1, 1, 0, 1]
    ])

    # Invert x, swap y, z
    chk_points = np.array([
        [0, 4, 0, 1],
        [-2, 0, 0, 1],
        [0, 0, 3, 1],
        [-1, 0, 1, 1]
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
        [0, 0, 4, 1],
        [2, 0, 0, 1],
        [0, 3, 0, 1],
        [1, 1, 0, 1]
    ])

    # Invert x, swap y, z
    cam_points = np.array([
        [0, 4, 0, 1],
        [-2, 0, 0, 1],
        [0, 0, 3, 1],
        [-1, 0, 1, 1]
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
        [1, 0, 0, 0],
        [0, 1, 0, 5],
        [0, 0, 1, 10],
        [0, 0, 0, 1]
    ])

    # Rotation around x axis by 90
    ref_rmtxX = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    # Rotation around z acis by 180
    ref_rmtxZ = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Output matrix
    ref_emtx = np.array([
        [-1, 0, 0, 0],
        [0, 0, -1, 5],
        [0, -1, 0, 10],
        [0, 0, 0, 1]
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

    v = 0.7071067811865476  # sin(45) and cos(45)
    chk_Rt = np.array([
        [1, 0, 0, 0],
        [0, v, -v, 0],
        [0, v, v, -5],
        [0, 0, 0, 1]])

    assert np.allclose(Rt, chk_Rt, atol=1e-5)

    # Test world to camera transform
    wc = cam.world_to_camera_matrix()
    Rtc = np.matmul(wc, Rt)
    chk_Rtc = np.array([
        [-1, 0, 0, 0],
        [0, v, v, -5],
        [0, v, -v, 0],
        [0, 0, 0, 1]])

    assert np.allclose(Rtc, chk_Rtc, atol=1e-5)


def test_intrinsic_matrix():
    """Intrinsic matrix"""

    # Hardcode parameters
    flen = 18.0
    sensor_world_width = 23.6
    sensor_pixel_res = 100.0

    # K
    sensor_pixel_width = sensor_world_width / sensor_pixel_res
    S = (flen * sensor_world_width) * sensor_pixel_width
    c = 0

    K = np.array([
        [S, 0, c, 0],
        [0, S, c, 0],
        [0, 0, 1, 0]])

    chk_K = cam.intrinsic_matrix(flen=flen, principle_point=(0, 0))

    assert np.allclose(K, chk_K, atol=1e-5)


def test_perspective_projection_transform():
    """Test perspective transformation"""
    v = 0.7071
    w = 1.4142
    m = w + v
    Rtc = np.array([
        [-1, 0, 0, 0],
        [0, v, v, -5],
        [0, v, -v, -10],
        [0, 0, 0, 1]
    ])
    K = np.array([
        [2, 0, 1, 0],
        [0, 2, 1, 0],
        [0, 0, 1, 0]
    ])

    # Check projection matrix
    P = np.matmul(K, Rtc)

    chk_P = np.array([
        [-2, v, -v, -10],
        [0, m, v, -20],
        [0, v, -v, -10]
    ])

    assert np.allclose(P, chk_P, atol=1e-5)

    # Check projection of segment left of the origin
    seg_left = np.array([
        [-5, 0, 0, 1],
        [-5, 5, 0, 1]
    ])

    chk_prj = np.array([
        [0, -20, -10],
        [3.5355, -9.3935, -6.4645]
    ])
    prj = np.matmul(P, seg_left.T).T

    assert np.allclose(prj, chk_prj, atol=1e-2)


def test_zbuffer():
    """Testing ordering of surfaces from zbuffering"""

    flen = 18
    heading = 0
    pitch = 45
    cam_posn = np.array([0, -5, 10])
    P = cam.projection_matrix(flen, heading, pitch, cam_posn)

    # Test ordering of points in center
    seg_center = np.array([
        [0, 0, 0, 1],  # pt_front
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
        [-5, 0, 0, 1],  # pt_front
        [-5, 5, 0, 1]  # pt_back
    ])

    prj_left = np.matmul(P, seg_left.T).T
    w = prj_center[:, 2]
    pt_front = prj_left[0] / w[0]
    pt_back = prj_left[1] / w[1]

    pp(pt_front)  # center
    pp(pt_back)  # to left

    # z coordinate of front point should be less then back
    assert pt_front[1] < pt_back[1]
    # x coordinate of front should be < back
    assert pt_front[0] > pt_back[0]


def test_bounding_box():
    # Calculate bbox from matrix of points
    ptmtx = np.array([
        [-10, 10, 0],
        [10, 5, 20],
        [10, 5, 20],
        [3, 12, -20]
    ])

    chk_bbox = np.array([
        [[-10, 12, -20], [10, 12, -20],
         [10, 5, -20], [-10, 5, -20]],
        [[-10, 5, 20], [10, 5, 20],
         [10, 12, 20], [-10, 12, 20]]
    ])

    bbox = cam._bounding_box(ptmtx)

    assert np.allclose(bbox, chk_bbox, atol=1e-10)


def test_single_plane_bounding_box():
    # Test bounding box when you get one surface
    ptmtx = np.array(
        [[-4, 0, 0], [4, 0, 0], [4, 0, 6], [-4, 0, 6]]
    )

    bot, top = cam._bounding_box(ptmtx)

    # Single line
    chkbot = np.array(
        [[-4, 0, 0],
         [4, 0, 0],
         [4, 0, 0],
         [-4, 0, 0]]
    )

    chktop = np.array(
        [[-4, 0, 6],
         [4, 0, 6],
         [4, 0, 6],
         [-4, 0, 6]]
    )

    assert np.allclose(chkbot, bot, 1e-10)
    assert np.allclose(chktop, top, 1e-10)


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

    flen = 18
    heading = 0
    pitch = 0
    cam_posn = np.array([0, -10, 5])
    P = cam.projection_matrix(flen, heading, pitch, cam_posn.copy())

    view_factor = cam.view_factor(P, srf) > 0.0
    assert cam._surface_normal(srf)[1] < 0.0
    assert view_factor

    view_factor = cam.view_factor(P, srf[::-1]) > 0.0
    assert cam._surface_normal(srf[::-1])[1] > 0.0
    assert not view_factor

    # Test view of simple horizontal plane with no rotation
    srf = np.array([
        [-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]
    ])

    flen = 18
    heading = 0
    pitch = 0
    cam_posn = np.array([0, -10, 5])
    P = cam.projection_matrix(flen, heading, pitch, cam_posn.copy())

    # Should be exactly perpendicular
    view_factor = cam.view_factor(P, srf) > 0.0
    assert cam._surface_normal(srf)[2] > 0.0
    assert view_factor

    view_factor = cam.view_factor(P, srf[::-1]) > 0.0
    assert cam._surface_normal(srf[::-1])[2] < 0.0
    assert not view_factor


def test_complex_view_factor():
    # Check view_hemisphere for each geoms
    bot_srf = np.array([
        [-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]
    ])

    top_srf = np.array([
        [-5, -5, 9], [5, -5, 9], [5, 5, 9], [-5, 5, 9]
    ])

    # P matrix see inner face of bottom and top faces
    flen = 18
    heading = r(45)
    pitch = r(-10)
    cam_posn = np.array([0, -15, 7])
    P = cam.projection_matrix(flen, heading, pitch, cam_posn.copy())

    view_factor = cam.view_factor(P, bot_srf) > 0.0
    assert view_factor

    view_factor = cam.view_factor(P, bot_srf[::-1]) > 0.0
    assert not view_factor

    # Look at top surface
    view_factor = cam.view_factor(P, top_srf) > 0.0
    assert not view_factor

    view_factor = cam.view_factor(P, top_srf[::-1]) > 0.0
    assert view_factor

    # Look at underside of bottom surface
    pitch = r(-55)
    P = cam.projection_matrix(flen, heading, pitch, cam_posn.copy())

    view_factor = cam.view_factor(P, bot_srf) > 0.0
    assert not view_factor

    view_factor = cam.view_factor(P, bot_srf[::-1]) > 0.0
    assert view_factor

    # P matrix see inner face of bottom and top faces
    pitch = r(35)
    P = cam.projection_matrix(flen, heading, pitch, cam_posn.copy())

    view_factor = cam.view_factor(P, bot_srf) > 0.0
    assert view_factor

    view_factor = cam.view_factor(P, bot_srf[::-1]) > 0.0
    assert not view_factor

    view_factor = cam.view_factor(P, top_srf) > 0.0
    assert view_factor

    view_factor = cam.view_factor(P, top_srf[::-1]) > 0.0
    assert not view_factor


def test_simple_snapshot():
    # Test projection of three simple surfaces

    # Define surfaces
    bot_srf = np.array(
        [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]])
    top_srf = np.array(
        [[-5, -5, 10], [5, -5, 10], [5, 5, 10], [-5, 5, 10]])
    vrt_srf = np.array(
        [[-4, 0, 0], [4, 0, 0], [4, 0, 6], [0, 0, 10], [-4, 0, 6]])
    geoms = [vrt_srf, top_srf, bot_srf]

    # Set camera parameters
    focal_len = 18
    heading = r(15)
    pitch = r(25)
    cam_point = np.array([0, -35, 4])

    # Ignore order for now.
    cam = Pincam(cam_point, heading, pitch, focal_len)
    xgeoms, depths = cam.project(cam.P, geoms)

    # Define the xgeoms. Not ordered
    fpath = os.path.join('tests', 'fixtures', 'simple_snapshot_surfaces.json')
    chk_xgeoms = load_fixture_array(fpath)

    # Assert
    for i, (xgeom, chk_xgeom) in enumerate(zip(xgeoms, chk_xgeoms)):
        assert np.allclose(xgeom, chk_xgeom, atol=1e-10)


def test_invert_extrinsic():
    """Invert extrinsic to reposition objects"""

    # Set camera parameters
    heading = r(45)
    pitch = r(0)
    cam_point = np.array([0, -35, 4])

    Rt = Pincam.extrinsic_matrix(heading, pitch, cam_point)
    Rt_copy = np.copy(Rt)

    # Rt:
    #   [[ 0.70710678, -0.70710678,  0.        ,  0.        ],
    #    [ 0.70710678,  0.70710678,  0.        , 35.        ],
    #    [ 0.        ,  0.        ,  1.        , -4.        ],
    #    [ 0.,          0. ,         0.        ,  1.        ]]

    # Make test matrices
    test_t = np.array(
        [[1., 0., 0., 0.],
         [0., 1., 0., -35.],
         [0., 0., 1., 4.],
         [0., 0., 0., 1.]])

    test_R = np.array(
        [[0.70710678, 0.70710678, 0., 0.],
         [-0.70710678, 0.70710678, 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.]])

    # test_t x test_R (Note that order is reversed)
    test_Rt = np.array(
        [[0.70710678, 0.70710678, 0., -24.7487373],
         [-0.70710678, 0.70710678, 0., -24.7487373],
         [0., 0., 1., 4.],
         [0., 0., 0., 1.]])

    # Test
    assert np.allclose(
        Pincam._invert_extrinsic_matrix_translation(Rt),
        test_t, atol=1e-5)

    assert np.allclose(
        Pincam._invert_extrinsic_matrix_rotation(Rt),
        test_R, atol=1e-10)

    print(Pincam.invert_extrinsic_matrix(Rt))

    assert np.allclose(
        Pincam.invert_extrinsic_matrix(Rt),
        test_Rt, atol=1e-10)

    # Ensure no mutations occured
    assert np.allclose(Rt, Rt_copy, atol=1e-10)


def test_project_camera_sensor_geometry():
    """Test the function to project geometry in 3D and reference sensor."""

    # Make camera
    focal_length = 20
    heading = r(0)
    pitch = r(10)
    cam_point = np.array([0, -45, 0])
    cam = Pincam(cam_point, heading, pitch, focal_length)

    # Get inverted extrinsic matrix
    iRt = cam.invert_extrinsic_matrix(cam.Rt)

    # Make camera points
    cam_pts = Pincam.project_camera_sensor_geometry(
        iRt, cam.sensor_plane_ptmtx_3d)
    test_cam_pts = np.array(
        [[-50., -52.99875777, -41.42621966],
         [50., -52.99875777, -41.42621966],
         [50., -35.63394, 57.05455565],
         [-50., -35.63394, 57.05455565],
         [-50., -52.99875777, -41.42621966]])

    # Test
    assert np.allclose(cam_pts, test_cam_pts, atol=7)


def test_raymtx():
    """Make raymtx from camera sensor plane"""

    # Make camera
    focal_length = 20
    heading = r(0)
    pitch = r(10)
    cam_point = np.array([0, -45, 0])
    cam = Pincam(cam_point, heading, pitch, focal_length)

    # Get and Rt camera sensor geometry
    plane = cam.sensor_plane_ptmtx_3d
    plane /= 10.0
    # plane:
    # [[-50.,   0., -50.],
    #  [ 50.,   0., -50.],
    #  [ 50.,   0.,  50.],
    #  [-50.,   0.,  50.],
    #  [-50.,   0., -50.]]

    # resolution of 3
    xx = np.array(
        [[-5, 0, 5],
         [-5, 0, 5],
         [-5, 0, 5]]) * 10.0

    yy = np.array(
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]) * 10.0

    zz = np.array(
        [[-5, -5, -5],
         [0, 0, 0],
         [5, 5, 5]]) * 10.0
    check_m = np.dstack([xx, yy, zz])

    # Test
    m = cam.ray_hit_matrix(cam.sensor_plane_ptmtx_3d, res=3)
    pp(m[:, :, 0])
    assert m.shape == (3, 3, 3)  # row, col, len(x,y,z)
    for i in range(3):
        assert np.allclose(m[:, :, i], check_m[:, :, i], atol=1e-10)

    # Test with higher res
    m = cam.ray_hit_matrix(cam.sensor_plane_ptmtx_3d, res=10)
    assert m.shape == (10, 10, 3)  # row, col, len(x,y,z)


def test_ray_hit_plane2():
    # Test ray plane interection

    # 2 x 2 square at y=2
    poly = np.array(
        [[-1., 2., -1.],
         [1., 2., -1.],
         [1., 2., 1.],
         [-1., 2., 1.]])
    poly_origin = [0, 2, 0]
    poly_norm = np.array([0, -1, 0])

    # Check straight intersection
    ray_pt = np.array([0, 0, 0])
    ray_dir = np.array([0, 1, 0])
    ipt = Pincam.ray_hit_plane2(ray_pt, ray_dir, poly_origin, poly_norm)
    print(ipt)
    assert np.allclose(ipt, np.array([0, 2, 0]), 1e-10)

    # Check no intersection
    ipt = Pincam.ray_hit_plane2(ray_pt, -ray_dir, poly_origin, poly_norm)
    print(ipt)
    assert ipt is None

    # Check reverse plane still hit
    ipt = Pincam.ray_hit_plane(ray_pt, ray_dir, poly_origin, -poly_norm)
    assert np.allclose(ipt, np.array([0, 2, 0]), atol=1e-10)


def test_rayhitpoly():
    # Test ray plane interection

    # 2 x 2 square at y=2
    poly = np.array(
        [[-1., 2., -1.],
         [1., 2., -1.],
         [1., 2., 1.],
         [-1., 2., 1.]])
    poly_origin = [0, 2, 0]
    poly_norm = np.array([0, -1, 0])

    # Check straight intersection
    ray_pt = np.array([0, 0, 0])
    ray_dir = np.array([0, 1, 0])
    ipt = Pincam.ray_hit_plane(ray_pt, ray_dir, poly_origin, poly_norm)
    assert np.allclose(ipt, np.array([0, 2, 0]), 1e-10)

    # Check no intersection
    ipt = Pincam.ray_hit_plane(ray_pt, -ray_dir, poly_origin, poly_norm)
    assert ipt is None

    # Check reverse plane still hit
    ipt = Pincam.ray_hit_plane(ray_pt, ray_dir, poly_origin, -poly_norm)
    assert np.allclose(ipt, np.array([0, 2, 0]), atol=1e-10)

    # Check intersection w/ poly
    ipt = Pincam.ray_hit_polygon(ray_pt, ray_dir, poly)
    assert np.allclose(ipt, np.array([0, 2, 0]), atol=1e-10)

    # Check hit plane but miss poly bounds
    ray_dir = np.array([0, 1, 2])  # 63.43 angle should  miss poly
    ipt = Pincam.ray_hit_plane(ray_pt, ray_dir, poly_origin, poly_norm)
    assert ipt is not None
    ipt = Pincam.ray_hit_polygon(ray_pt, ray_dir, poly)
    assert ipt is None


def test_image_matrix():
    """Test image matrix."""

    poly_front = np.array(
        [[-5, 0, 0], [5, 0, 0], [5, 0, 5], [-5, 0, 5]])
    poly_back = np.array(
        [[-5, 2, 0], [5, 2, 0], [5, 2, 5], [-5, 2, 5]])
    ptmtx = [poly_front, poly_back]

    # Make camera
    focal_length = 20
    heading = r(0)
    pitch = r(10)
    cam_point = np.array([0, -30, 0])
    cam = Pincam(cam_point, heading, pitch, focal_length)

    imgs = cam.image_matrix(ptmtx, inches=8, dpi=15)

    assert len(imgs) == 2
    assert isinstance(imgs[0], np.ndarray)
    assert isinstance(imgs[1], np.ndarray)

    img = imgs[0]

    assert img.shape[0] == 120  # row pixel dims
    assert img.shape[1] == 120  # col pixel dims
    assert img.shape[2] == 3  # rgb channels

    # 100 x 100 default
    imgs = cam.image_matrix(ptmtx, inches=10, dpi=10)

    assert len(imgs) == 2
    assert isinstance(imgs[0], np.ndarray)
    assert isinstance(imgs[1], np.ndarray)

    img = imgs[0]

    assert img.shape[0] == 100  # row pixel dims
    assert img.shape[1] == 100  # col pixel dims
    assert img.shape[2] == 3  # rgb channels


def test_depth_buffer():
    """Test building depth buffer."""

    poly_front = np.array(
        [[-5, 0, 0], [5, 0, 0], [5, 0, 5], [-5, 0, 5]])
    poly_back = np.array(
        [[-5, 2, 0], [5, 2, 0], [5, 2, 5], [-5, 2, 5]])

    # Make camera
    focal_length = 20
    heading = r(0)
    pitch = r(10)
    cam_point = np.array([0, -10, 0])
    cam = Pincam(cam_point, heading, pitch, focal_length)

    # Test
    ptmtx = [poly_front, poly_back]
    test_depths = [1, 0]
    _, _depths = cam.project(cam.P, ptmtx)
    depths, _ = cam.depth_buffer(ptmtx, _depths, res=25)
    assert len(depths) == 2
    assert np.allclose(depths, test_depths, atol=1e-10)

    # Test 2
    ptmtx = [poly_back, poly_front]
    test_depths = [0, 1]
    _, _depths = cam.project(cam.P, ptmtx)
    depths, _ = cam.depth_buffer(ptmtx, _depths, res=25)
    assert len(depths) == 2
    assert np.allclose(depths, test_depths, atol=1e-10)

    # Why does this fail
    ptmtx = [poly_front, poly_back]
    test_depths = [1, 0]
    _depths = [0, 1]
    depths, _ = cam.depth_buffer(ptmtx, _depths, res=25)
    assert len(depths) == 2
    assert np.allclose(depths, test_depths, atol=1e-10)


def test_reorder_depths():
    """Test reordering depth buffer"""

    depths = [1, 2, 3, 4]
    cur_geo = 2
    min_geo = 1

    test_depths = [2, 1, 3, 4]
    depths = Pincam.reorder_depths(depths, cur_geo, min_geo)

    assert np.allclose(test_depths, depths, atol=1e-10)

    # Now assume 3 is min
    depths = [2, 1, 3, 4]
    cur_geo = 3
    min_geo = 2

    test_depths = [3, 2, 1, 4]
    depths = Pincam.reorder_depths(depths, cur_geo, min_geo)

    assert np.allclose(test_depths, depths, atol=1e-10)
