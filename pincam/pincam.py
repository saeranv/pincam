import numpy as np
from .matrix_utils2 import MatrixUtils2 as mu
from pprint import pprint as pp


def p2e(p):
    # Matrix of projective to euclidian
    w = p[2, :] # row of w = y depth
    return (p / w)[0:2, :].T


def ortho_p2e(p):
    # Matrix of projective to euclidian
    w = 10
    return (p / w)[0:2, :].T


def e2p(e):
    # Matrix of euclidean to projective
    # Converts to column vectors
    return np.insert(e, 3, 1, 1).T


def world_to_camera_matrix():
    """Changes coordinate system from cartesian world to camera coordinates.

    Matrix flips y, z.
    """

    return np.array([
        [-1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1]
    ])


def camera_to_world_matrix():
    """Use same matrix to flip the matrix
    """

    return world_to_camera_matrix()

def extrinsic_matrix(heading, pitch, cam_posn):
    """
    TBD
    """
    # Init parameters
    origin = np.array([0, 0, 0])
    cam_posn = -1 * cam_posn.copy()
    #heading *= -1
    #pitch *= -1

    # TODO: Invert all of this
    # Make Rz matrix
    z_axis = np.array([0, 0, 1])
    Rz = mu.xform_rotation_matrix(origin, z_axis, heading)

    # Make Rx matrix
    x_axis = np.array([1, 0, 0])
    Rx = mu.xform_rotation_matrix(origin, x_axis, pitch)

    # Make translation matrix
    T = mu.xform_translation_matrix(cam_posn)

    # Multiply
    Rt = mu.matmul_xforms([Rz, Rx, T])

    return Rt


def intrinsic_matrix(fov, flen=0.1, principle_point=(0, 0)):
    """

    # http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf

    x' = (X * f / sx) + (Z * px)
    y' = (Y * f / sy) + (Z * py)
    z' = Z

    Then:
    u = x' / z' = ((X * f / sx) + (Z * px)) / Z
    v = y' / z' = ((Y * f / sy) + (Z * py)) / Z

    Args:
        f = focal length
        sx, sy = sensor resolution
        px, py = principle point coordinates

    Returns:
        3 x 4 K matrix
    """


    # Assume square aspect ratio for now
    #sensor_world_width = 0.5
    min_flen, max_flen = 0.1, 5
    delta_flen = max_flen - min_flen
    assert flen <= max_flen and flen >= min_flen

    #max_sensor_world_width = 2 * np.tan(fov / 2.0) * max_flen
    #min_sensor_world_width = 2 * np.tan(fov / 2.0) * min_flen
    #delta_sensor_world_width = max_sensor_world_width - min_sensor_world_width
    sensor_world_width = 2 * np.tan(fov / 2.0) * flen
    sensor_pixel_width = 100
    sensor_pixel_width = sensor_pixel_width * flen / delta_flen
    # Multiply this by world coords to get pixel coords
    #pixel_conv_factor = sensor_pixel_width #/ sensor_world_width
    px, py = principle_point
    #Sx = Sy = (flen * pixel_conv_factor) #/ sensor_pixel_width
    Sx = Sy = flen / sensor_world_width * sensor_pixel_width

    K = np.array([
        [Sx,  0,  px,  0],
        [ 0, Sy,  py,  0],
        [ 0,  0,  1,   0]])

    return K


def projection_matrix(fov, focal_length, heading, pitch, cam_posn):
    """
    Create transformation matrix which combines rotation along z, x axis, translation

    """
    Rt = extrinsic_matrix(heading, pitch, cam_posn)
    wc = world_to_camera_matrix()
    Rtc = np.matmul(wc, Rt)
    K = intrinsic_matrix(fov, flen=focal_length)
    R = np.eye(4, 4)
    R[:3,:3] = Rt[:3,:3]
    return np.matmul(K, Rtc), np.matmul(K, Rtc)


def stack(geometries):
    """
    TBD
    """
    ptnums = [np.shape(geometry)[0] for geometry in geometries]
    idx = np.cumsum(ptnums[:-1]) # list of end index for every geometry
    stacked = np.concatenate(geometries)

    return stacked, idx


def sensor_bounds(sensor_pixel_width=100):
    """
    TBD
    """

    pw = sensor_pixel_width / 2.
    bounds = np.array([
        [-pw, -pw], [pw, -pw], [pw, pw], [-pw, pw], [-pw, -pw]
    ])

    return bounds


def _bounding_box(geometries):
    """Returns bounding box of geometry matrix"""
    # Get extents
    minx = np.min(geometries[:, 0])
    miny = np.min(geometries[:, 1])
    minz = np.min(geometries[:, 2])
    maxx = np.max(geometries[:, 0])
    maxy = np.max(geometries[:, 1])
    maxz = np.max(geometries[:, 2])

    # Construct bbox bottom and top surfaces
    bbox = np.array([
        [[minx, maxy, minz], [maxx, maxy, minz],
         [maxx, miny, minz], [minx, miny, minz]],
        [[minx, miny, maxz], [maxx, miny, maxz],
         [maxx, maxy, maxz], [minx, maxy, maxz]],
    ])

    return bbox


def _surface_normal(surface):
    """Calculate normal from surface"""

    vec1 = surface[1] - surface[0]
    vec2 = surface[2] - surface[1]

    normal = np.cross(vec1, vec2)

    if np.abs(np.sum(normal)) < 1e-10:
        return False

    return normal / np.linalg.norm(normal)


def view_factor(P, surface):
    """
    Returns boolean corresponding to if top, bottom bounding box is within
    view hemisphere.

    view_factor = view_factor > 0.0

    """
    # Camera coordinates in world coordinates
    cam_point = np.array([0, 1, 0]) #cam_posn

    # TODO: Convert this into a geometry to view_frustrum method
    # TODO: insert view_frustrum method in p2e

    wmtx = camera_to_world_matrix()
    xsurface = np.matmul(P, e2p(surface))
    w = xsurface[2, :]
    xsurface = (xsurface / w).T # squish x,y into camera view frustrum
    xsurface[:,2] = w
    xsurface = np.insert(xsurface, 3, 1, 1)
    # Convert back to world coordinates
    xsurface = np.matmul(wmtx, xsurface.T).T
    xsurface = xsurface[:,:3]
    #V0 = surface[0]
    N = _surface_normal(xsurface)
    #view_factor = not np.dot(V0 - P, N) >= 0.0
    if isinstance(N, np.ndarray):
        return np.dot(-cam_point, N)
    else:
        print('bbox fail for:', surface)
        return 1.0


def _view_bounding_extents(P, cam_posn, geometries):
    # Determine which faces are viewable in bounding box
    flattened_geometries = np.array(
        [point for geometry in geometries for point in geometry])
    srfs = _bounding_box(flattened_geometries)
    # Since bounding box orients normals towards outside of box, flip surfaces
    srf_bot, srf_top = srfs[0][::-1], srfs[1][::-1]
    #print(srf_bot, srf_top)

    # Check if inside bbox faces can be seen by camera
    view_bot_factor = view_factor(P, srf_bot)
    view_bot = view_bot_factor > 0.0
    view_top_factor = view_factor(P, srf_top)
    view_top = view_top_factor > 0.0
    #print('--')
    return (view_bot, view_top), (view_bot_factor, view_top_factor)


def project_by_z(P, Rtc, cam_posn, geometries, orth=False):

    def _helper_project(P, cam_posn, _grouped_by_z):
        # Project
        proj_geoms = []
        for geoms in _grouped_by_z:
            pgeoms = project(P, cam_posn, geoms.T[0], ortho=False)
            pgeoms = [np.array(pgeom) for pgeom in pgeoms]
            proj_geoms.extend(pgeoms)
        return proj_geoms

    # Get z order
    zlst = [np.mean(g[:, 2]) for g in geometries]

    foo = lambda k: k[:, 1]
    grouped_by_z, _ = mu.groupby(np.array([geometries, zlst]).T,
                                 tol=1e-10, axis_lambda=foo)

    view_data = _view_bounding_extents(Rtc, cam_posn, geometries)
    view_bot, view_top = view_data[0]
    view_bot_factor, view_top_factor = view_data[1]

    # The closer view factor is to 1, more in view
    #print('bot', view_bot, view_bot_factor)
    #print('top', view_top, view_top_factor)

    if view_bot and view_top:
        t, b, c = [], [], []
        for geoms in grouped_by_z:
            _geoms = geoms.T[0]
            _view, _viewf = _view_bounding_extents(P, cam_posn, _geoms)
            _view_bot, _view_top = _view
            #print(geoms)
            #print(_view_bot, _view_top)
            #print(_viewf)
            #print('--')
            #pgeoms = _helper_project(P, cam_posn, geoms)
            #proj_geoms.extend(pgeoms)
            if _view_top and _view_bot:
                c.append(geoms)
            elif _view_top:
                t.append(geoms)
            else:
                b.append(geoms)

        b = b[::-1]
        if view_top_factor > view_bot_factor:
            # if we see more of the top, start with that
            grouped_by_z = t + b + c
        else:
            grouped_by_z = b + t + c
        return _helper_project(P, cam_posn, grouped_by_z)

    elif view_bot:
        return _helper_project(P, cam_posn, grouped_by_z)

    elif view_top:
        grouped_by_z = reversed(grouped_by_z)
        return _helper_project(P, cam_posn, grouped_by_z)



def project(P, cam_posn, geometries, ortho=False):
    """
    TBD
    """
    # Stack arrays for more efficient matrix multiplication
    ptmtx, idx = stack(geometries)

    # MuLtiply geometries by P matrix
    ptmtx = e2p(ptmtx)
    xptmtx = np.matmul(P, ptmtx)

    furthest_depths = [max(warr) for warr in np.split(xptmtx[2], idx)]

    ordered_depths = np.argsort(furthest_depths)[::-1]

    if ortho:
        xptmtx = ortho_p2e(xptmtx)
    else:
        xptmtx = p2e(xptmtx)

    # Split and sort by z buffer
    xgeometries = np.array(np.split(xptmtx, idx))

    return xgeometries[ordered_depths].tolist()

