import numpy as np
from .matrix_utils2 import MatrixUtils2 as mu
from pprint import pprint as pp


def p2e(p):
    """Matrix of projective to euclidian.

    For ortho:
    #w = 10
    #return (p / w)[0:2, :].T
    """
    w = p[2, :] # row of w = y depth
    return (p / w)[0:2, :].T


def e2p(e):
    """Matrix of euclidean to projective
    Converts to column vectors
    """
    return np.insert(e, 3, 1, 1).T


class Pincam(object):
    """Lightweight pinhole camera.

    Args:
        fov: Horizontal and vertical camera Field of View (FOV) in radians.
        heading: Rotation around z-axis in radians.
        pitch: Rotation around x-axis in radians.
        cam_point: Camera location in 3d cartesian coordinates, as a numpy array.

    Properties:
        * Rt
        * world_to_camera_matrix
        * K
        * P
        * frame
    """

    RAD35 = 35.0 / 180.0 * np.pi
    RAD45 = 45.0 / 180.0 * np.pi
    DEFAULT_PIXEL_RESOLUTION = 100.0
    DEFAULT_SENSOR_WORLD_WIDTH = 23.6
    DEFAULT_MIN_FOCAL_LENGTH = 18
    DEFAULT_MAX_FOCAL_LENGTH = 70

    def __init__(self, cam_point, heading=RAD45, pitch=RAD45, focal_length=18):
        """Initialize Pincam"""

        # Camera position
        self.cam_point = cam_point

        # Camera parameters
        # TODO: reset P, Rtc, K when we change these
        self.focal_length = focal_length
        self.heading = heading
        self.pitch = pitch

        # Camera matrices
        self._P = None

    def __repr__(self):
        return 'Camera parameters:' + \
            '\nCamera point: {}'.format(self.cam_point.tolist()) + \
            '\nFocal length: {}mm'.format(self.focal_length) + \
            '\nHeading: {} deg'.format(self.heading / np.pi * 180.0) + \
            '\nPitch: {} deg'.format(self.pitch / np.pi * 180.0)

    @property
    def sensor_plane_ptmtx_2d(self):
        """Get camera sensor_panel"""

        pw = 50.0  #self.DEFAULT_SENSOR_WORLD_WIDTH
        return np.array(
            [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * pw

    @property
    def sensor_plane_ptmtx_3d(self):
        """Get camera sensor_panel"""

        pw = 50.0  #self.DEFAULT_SENSOR_WORLD_WIDTH / 2.
        return np.array(
            [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1], [-1, 0, -1]]) * pw


    @property
    def P(self):
        """Get projection matrix P."""
        if True:#if self._P is None:
            self._P = Pincam.projection_matrix(
                self.focal_length, self.heading, self.pitch, self.cam_point)
        return self._P

    @staticmethod
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

    @staticmethod
    def camera_to_world_matrix():
        """Use same matrix to flip the matrix
        """

        return Pincam.world_to_camera_matrix()

    @staticmethod
    def extrinsic_matrix(heading, pitch, cam_posn):
        """
        Affine transformation (combination of linear transformation and
        translation) are linear transforms where the origin does not
        neccessarily map to origin.
        Ref: http://graphics.cs.cmu.edu/courses/15-463/2006_fall/www/Lectures/warping.pdf
        """
        # Init parameters
        origin = np.array([0, 0, 0])
        cam_posn = -1 * cam_posn.copy()

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

    @staticmethod
    def intrinsic_matrix(flen=18, principle_point=(0, 0)):
        """
        # http://www.cse.psu.edu/~rtc12/CSE486/lecture11.pdf
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

        # Sony DSLR A-100
        # Ref: https://en.wikipedia.org/wiki/Sony_Alpha_100
        #focal_length = 18 # 18 - 70
        #sensor_width = 23.6
        #sensor_height = 23.6 #15.6
        #pixel_num_width = 3872
        #pixel_num_height = 3872 #2592

        # Assume square aspect ratio for now
        #sensor_world_width = 0.5
        # TODO: convert the mm into m for flen and sensor dims
        min_flen = Pincam.DEFAULT_MIN_FOCAL_LENGTH  # convert to m
        max_flen = Pincam.DEFAULT_MAX_FOCAL_LENGTH
        assert (flen <= max_flen) and (flen >= min_flen), \
            '{} >= focal length >= {}'.format(min_flen, max_flen)

        #max_sensor_world_width = 2 * np.tan(fov / 2.0) * max_flen
        #min_sensor_world_width = 2 * np.tan(fov / 2.0) * min_flen
        #delta_sensor_world_width = max_sensor_world_width - min_sensor_world_width
        sensor_world_width = Pincam.DEFAULT_SENSOR_WORLD_WIDTH
        fov = np.arctan(flen / sensor_world_width)
        #sensor_world_width = 2 * np.tan(fov / 2.0) * flen
        #sensor_pixel_width = 100
        #sensor_pixel_width = sensor_pixel_width * flen / delta_flen
        sensor_pixel_res = Pincam.DEFAULT_PIXEL_RESOLUTION
        sensor_pixel_width = sensor_world_width / sensor_pixel_res
        # Multiply this by world coords to get pixel coords
        #pixel_conv_factor = sensor_pixel_width #/ sensor_world_width
        px, py = principle_point
        #Sx = Sy = (flen * pixel_conv_factor) #/ sensor_pixel_width
        #Sx = flen / sensor_world_width * sensor_pixel_width
        Sx = (flen * sensor_world_width) * sensor_pixel_width
        Sy = Sx
        K = np.array([
            [Sx,  0,  px,  0],
            [ 0, Sy,  py,  0],
            [ 0,  0,  1,   0]])

        return K

    @staticmethod
    def _invert_extrinsic_matrix_translation(Rt):
        """Invert translation in extrinsic matrix"""

        # Invert translation is negative vector
        _Rt = np.eye(4) # Make new matrix to avoid mutations
        t = Rt[:3, 3]
        _Rt[:3, 3] = -t

        return _Rt

    @staticmethod
    def _invert_extrinsic_matrix_rotation(Rt):
        """Invert rotation in extrinsic matrix"""

        # Invert rotation matrix is it's transpose
        _Rt = np.eye(4) # Make new matrix to avoid mutations
        R = Rt[:3, :3]
        _Rt[:3, :3] = R.T # transpose

        return _Rt

    @staticmethod
    def invert_extrinsic_matrix(Rt):
        """Invert rotation and translation in extrinsic matrix

        Order of transformations is important. First inverse translation
        and then inverse rotation.
        """

        it = Pincam._invert_extrinsic_matrix_translation(Rt)
        iR = Pincam._invert_extrinsic_matrix_rotation(Rt)

        return mu.matmul_xforms([it, iR])

    @staticmethod
    def projection_matrix(focal_length, heading, pitch, cam_point):
        """
        Transformation matrix which combines rotation along z, x axis, translation.
        """
        Rt = Pincam.extrinsic_matrix(heading, pitch, cam_point)
        wc = Pincam.world_to_camera_matrix()
        Rtc = np.matmul(wc, Rt)
        K = Pincam.intrinsic_matrix(flen=focal_length)
        R = np.eye(4, 4)
        R[:3,:3] = Rt[:3,:3]

        return np.matmul(K, Rtc)

    @staticmethod
    def stack(geometries):
        """
        TBD
        """
        ptnums = [np.shape(geometry)[0] for geometry in geometries]
        idx = np.cumsum(ptnums[:-1]) # list of end index for every geometry
        stacked = np.concatenate(geometries)

        return stacked, idx

    @staticmethod
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

    @staticmethod
    def _surface_normal(surface):
        """Calculate normal from surface"""

        vec1 = surface[1] - surface[0]
        vec2 = surface[2] - surface[1]

        normal = np.cross(vec1, vec2)

        if np.abs(np.sum(normal)) < 1e-10:
            return False

        return normal / np.linalg.norm(normal)

    @staticmethod
    def view_factor(P, surface):
        """Calculate the view factor (the projection of the surface to the principle point).

        Note, the surface face is in view if view_factor > 0.0, which assumes view has a
        field of view of 180 degrees.

        Args:
            P: Projection matrix
            surface: Point matrix of a surface in world coordinates.

        Returns:
            View factor for surface.
        """
        # Camera position in world coordinates
        cam_point = np.array([0, 1, 0])

        xsurface = Pincam.project3d(P, surface)

        # Get normal from surface squished into view frustum
        N = Pincam._surface_normal(xsurface)

        # Temporary hack b/c certain normals are 0 here
        if isinstance(N, np.ndarray):
            # view_factor = np.dot(-cam_ray, N) // Note we flip cam_ray for projection
            # view = view_factor >= 0.0
            return np.dot(-cam_point, N)
        else:
            return 1.0

    @staticmethod
    def _view_bounding_extents(P, cam_posn, geometries):
        # Determine which faces are viewable in bounding box
        flattened_geometries = np.array(
            [point for geometry in geometries for point in geometry])
        srfs = Pincam._bounding_box(flattened_geometries)

        # Since bounding box orients normals towards outside of box, flip surfaces
        srf_bot, srf_top = srfs[0][::-1], srfs[1][::-1]

        # Check if inside bbox faces can be seen by camera
        view_bot_factor = Pincam.view_factor(P, srf_bot)
        view_bot = view_bot_factor > 0.0
        view_top_factor = Pincam.view_factor(P, srf_top)
        view_top = view_top_factor > 0.0

        return (view_bot, view_top), (view_bot_factor, view_top_factor)

    @staticmethod
    def project3d(P, surface):
        """This is a 3D projection transformation.

        Squish orthogonal geometries into 3d view frustum.

        Args:
            surface: n x 3 point matrix of orthogonal surface

        Returns:
            surface: n x 3 point matrix of distorted perspective surface
        """
        # 3d ptmtx n x 3 ortho geometry
        # Affine transformation of 3d surface to 2d planar projection
        xsurface = np.matmul(P, e2p(surface))  # n x 4 matrix

        # Dividing x, y by depth shrinks surface proportional to depth in view frustum
        w = xsurface[2, :]  # Column vector of depths
        xsurface = (xsurface / w).T

        # Convert from 2d to 3d world coordinates
        xsurface[:, 2] = w  # Add depth information into 3rd column to make 3d
        xsurface = np.insert(xsurface, 3, 1, 1)  # n x 4 matrix of homogenous coordinates
        xsurface = np.matmul(Pincam.camera_to_world_matrix(), xsurface.T).T

        # 3d ptmtx n x 3 perspective geometry
        return xsurface[:, :3]

    @staticmethod
    def project(P, cam_posn, geometries):
        """
        TBD
        """

        # Stack arrays for more efficient matrix multiplication
        ptmtx, idx = Pincam.stack(geometries)

        # MuLtiply geometries by P matrix
        ptmtx = e2p(ptmtx)
        xptmtx = np.matmul(P, ptmtx)

        furthest_depths = [max(warr) for warr in np.split(xptmtx[2], idx)]

        ordered_depths = np.argsort(furthest_depths)[::-1]

        xptmtx = p2e(xptmtx)

        # Split and sort by z buffer
        xgeometries = np.array(np.split(xptmtx, idx))

        return xgeometries[ordered_depths].tolist()

    def project_by_z(self, geometries, ortho=False):
        #TODO: Deprecate
        def _helper_project(P, cam_posn, _grouped_by_z):
            # Project
            proj_geoms = []
            for geoms in _grouped_by_z:
                pgeoms = Pincam.project(P, cam_posn, geoms.T[0])
                pgeoms = [np.array(pgeom) for pgeom in pgeoms]
                proj_geoms.extend(pgeoms)
            return proj_geoms

        P = self.P
        cam_posn = self.cam_point

        # Get z order
        zlst = [np.mean(g[:, 2]) for g in geometries]

        foo = lambda k: k[:, 1]
        grouped_by_z, _ = mu.groupby(np.array([geometries, zlst]).T,
                                    tol=1e-10, axis_lambda=foo)

        view_data = Pincam._view_bounding_extents(P, cam_posn, geometries)
        view_bot, view_top = view_data[0]
        view_bot_factor, view_top_factor = view_data[1]

        # The closer view factor is to 1, more in view
        #print('bot', view_bot, view_bot_factor)
        #print('top', view_top, view_top_factor)

        if view_bot and view_top:
            t, b, c = [], [], []
            for geoms in grouped_by_z:
                _geoms = geoms.T[0]
                _view, _viewf = self._view_bounding_extents(P, cam_posn, _geoms)
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

            #return _helper_project(P, cam_posn, grouped_by_z)

        elif view_bot:
            pass
            #return _helper_project(P, cam_posn, grouped_by_z)

        elif view_top:
            grouped_by_z = reversed(grouped_by_z)
            #return _helper_project(P, cam_posn, grouped_by_z)
        else:
            print('Nothing in view. Check if camera is too close too object.')

        return _helper_project(P, cam_posn, grouped_by_z)

    def view_frustum_geometry(self, ptmtx, show_cam=True):
        """View the geometries in the view frustrum

        Args:
            ptmtx: List of surfaces as numpy array of points.
            show_ref_cam: Show the camera that is 'viewing' the geometry (Default: True).

        Returns:
            List of surfaces projected in 3d, with reference camera
                as surface.

        """
        # Project sensor, surface geometries in 3d
        _ptmtx = [Pincam.project3d(self.P, pts) for pts in ptmtx]

        # Invert the affine transformations (rotation, translation)
        Rt = Pincam.extrinsic_matrix(self.heading, self.pitch, self.cam_point)
        it = Pincam._invert_extrinsic_matrix_translation(Rt)
        iR = Pincam._invert_extrinsic_matrix_rotation(Rt)
        iRt = mu.matmul_xforms([it, iR])

        _ptmtx = [np.insert(srf, 3, 1, 1) for srf in _ptmtx]

        # Apply inverse transformations
        _ptmtx = [np.matmul(iRt, srf.T).T for srf in _ptmtx]

        if show_cam:
            cam_pts = np.insert(self.sensor_plane_ptmtx_3d, 3, 1, 1)
            iR = Pincam._invert_extrinsic_matrix_rotation(Rt)
            it = Pincam._invert_extrinsic_matrix_translation(Rt)
            iRt = mu.matmul_xforms([it, iR])
            cam_pts = np.matmul(iRt, cam_pts.T).T
            _ptmtx += [cam_pts]  # add camera sensor

        _ptmtx = [srf[:, :3] for srf in _ptmtx]

        return _ptmtx

    def to_gpd_geometry(self, ptmtx):
        """Project geometries to 3d from geopandas dataframe

        Args:
            df: GeoPandas.Dataframe.

        Returns:
            Dataframe with geometry.
        """
        projected_ptmtx = self.project_by_z(ptmtx, ortho=False)

        return [mu.shapely_from_srf3d(_ptmtx) for _ptmtx in projected_ptmtx]
