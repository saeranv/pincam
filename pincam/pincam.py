import numpy as np
from ladybug_geometry.geometry3d import Point3D, Vector3D, Ray3D, Plane, Face3D
from ladybug_geometry.geometry2d import Point2D
from pprint import pprint as pp
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import geometry
from .matrix import xform_rotation_matrix, xform_translation_matrix, matmul_xforms
from PIL import Image


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
    def P(self):
        """Get projection matrix P."""
        return self.projection_matrix(
            self.focal_length, self.heading, self.pitch, self.cam_point)

    @property
    def Rt(self):
        """Get extrinsic matrix"""
        return self.extrinsic_matrix(self.heading, self.pitch, self.cam_point)

    @property
    def sensor_plane_ptmtx_2d(self):
        """Get camera sensor_panel"""

        pw = 50.0  # self.DEFAULT_SENSOR_WORLD_WIDTH
        return np.array(
            [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * pw

    @property
    def sensor_plane_ptmtx_3d(self):
        """Get camera sensor_panel"""

        pw = 50.0  # self.DEFAULT_SENSOR_WORLD_WIDTH / 2.
        return np.array(
            [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1], [-1, 0, -1]]) * pw

    @staticmethod
    def p2e(p, ortho=False):
        """Matrix of projective to euclidian.

        For ortho:
        #w = 10
        #return (p / w)[0:2, :].T
        """
        w = 1.0
        if not ortho:
            w = p[2, :]  # row of w = y depth
        return (p / w)[0:2, :].T

    @staticmethod
    def e2p(e):
        """Matrix of euclidean to projective
        Converts to column vectors
        """
        return np.insert(e, 3, 1, 1).T


    @staticmethod
    def to_poly_sh(poly_np):
        """ Srf cab be 3dm shapely init will automatically remove extra dims"""
        return geometry.Polygon(poly_np[:, :2])


    @staticmethod
    def world_to_camera_matrix():
        """Changes coordinate system from cartesian world to camera coordinates.

        Matrix flips y, z.
        """

        return np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def camera_to_world_matrix():
        """Use same matrix to flip the matrix
        """

        return Pincam.world_to_camera_matrix()

    @staticmethod
    def extrinsic_matrix(heading, pitch, cam_point):
        """
        Affine transformation (combination of linear transformation and
        translation) are linear transforms where the origin does not
        neccessarily map to origin.
        Ref: http://graphics.cs.cmu.edu/courses/15-463/2006_fall/www/Lectures/warping.pdf
        """
        # Init parameters
        origin = np.array([0, 0, 0])
        cam_posn = -1 * cam_point.copy()

        # TODO: Invert all of this
        # Make Rz matrix
        z_axis = np.array([0, 0, 1])
        Rz = xform_rotation_matrix(origin, z_axis, heading)

        # Make Rx matrix
        x_axis = np.array([1, 0, 0])
        Rx = xform_rotation_matrix(origin, x_axis, pitch)

        # Make translation matrix
        T = xform_translation_matrix(cam_posn)

        # Multiply
        Rt = matmul_xforms([Rz, Rx, T])

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
        # focal_length = 18 # 18 - 70
        #sensor_width = 23.6
        # sensor_height = 23.6 #15.6
        #pixel_num_width = 3872
        # pixel_num_height = 3872 #2592

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
        # pixel_conv_factor = sensor_pixel_width #/ sensor_world_width
        px, py = principle_point
        # Sx = Sy = (flen * pixel_conv_factor) #/ sensor_pixel_width
        #Sx = flen / sensor_world_width * sensor_pixel_width
        Sx = (flen * sensor_world_width) * sensor_pixel_width
        Sy = Sx
        K = np.array([
            [Sx, 0, px, 0],
            [0, Sy, py, 0],
            [0, 0, 1, 0]])

        return K

    @staticmethod
    def _invert_extrinsic_matrix_translation(Rt):
        """Invert translation in extrinsic matrix"""

        # Invert translation is negative vector
        _Rt = np.eye(4)  # Make new matrix to avoid mutations
        t = Rt[:3, 3]
        _Rt[:3, 3] = -t

        return _Rt

    @staticmethod
    def _invert_extrinsic_matrix_rotation(Rt):
        """Invert rotation in extrinsic matrix"""

        # Invert rotation matrix is it's transpose
        _Rt = np.eye(4)  # Make new matrix to avoid mutations
        R = Rt[:3, :3]
        _Rt[:3, :3] = R.T  # transpose

        return _Rt

    @staticmethod
    def invert_extrinsic_matrix(Rt):
        """Invert rotation and translation in extrinsic matrix

        Order of transformations is important. First inverse translation
        and then inverse rotation.
        """

        it = Pincam._invert_extrinsic_matrix_translation(Rt)
        iR = Pincam._invert_extrinsic_matrix_rotation(Rt)

        return matmul_xforms([it, iR])

    @staticmethod
    def projection_matrix(focal_length, heading, pitch, cam_point):
        """
        Transformation matrix which combines rotation along z, x axis, translation.

        Args:
            focal_length: focal length between 18 and 35 (inclusive).
            heading: heading in radians.
            pitch: pitch in radians.
            cam_point: camera location.

        Return:
            Projection matrix.
        """
        Rt = Pincam.extrinsic_matrix(heading, pitch, cam_point.copy())
        wc = Pincam.world_to_camera_matrix()
        Rtc = np.matmul(wc, Rt)
        K = Pincam.intrinsic_matrix(flen=focal_length)
        R = np.eye(4, 4)
        R[:3, :3] = Rt[:3, :3]

        return np.matmul(K, Rtc)

    @staticmethod
    def stack(geometries):
        """
        TBD
        """
        ptnums = [np.shape(geometry)[0] for geometry in geometries]
        idx = np.cumsum(ptnums[:-1])  # list of end index for every geometry
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
        xsurface = np.matmul(P, Pincam.e2p(surface))  # n x 4 matrix

        # Dividing x, y by depth shrinks surface proportional to depth in view frustum
        w = xsurface[2, :]  # Column vector of depths
        xsurface = (xsurface / w).T

        # Convert from 2d to 3d world coordinates
        xsurface[:, 2] = w  # Add depth information into 3rd column to make 3d
        # n x 4 matrix of homogenous coordinates
        xsurface = np.insert(xsurface, 3, 1, 1)
        xsurface = np.matmul(Pincam.camera_to_world_matrix(), xsurface.T).T

        # 3d ptmtx n x 3 perspective geometry
        return xsurface[:, :3]

    @staticmethod
    def project(P, geometries, ortho=False, depth_by_mean=True):
        """
        TBD
        """

        # Stack arrays for more efficient matrix multiplication
        ptmtx, idx = Pincam.stack(geometries)

        # MuLtiply geometries by P matrix
        ptmtx = Pincam.e2p(ptmtx)
        xptmtx = np.matmul(P, ptmtx)

        if depth_by_mean:
            depth_select_fx = lambda w: np.mean(w)
        else:
            depth_select_fx = lambda w: np.min(w)

        furthest_depths = [depth_select_fx(warr) for warr in np.split(xptmtx[2], idx)]

        ordered_depths = np.argsort(furthest_depths)
        xptmtx = Pincam.p2e(xptmtx, ortho=ortho)

        # Split and sort by z buffer
        xgeometries = np.split(xptmtx, idx)

        return xgeometries, ordered_depths[::-1].tolist()


    @staticmethod
    def ray_hit_matrix(sensor_plane_3d, res=10):
        """Ray hit matrix

        Args:
            res: Integer with integer square root.
        """
        p = sensor_plane_3d
        res -= 1
        # Principal point at origin at 0,0,0
        # Sensor plane is square matrix
        minb, maxb = np.min(p), np.max(p)
        step = (maxb - minb) / (res + 1)

        xx, zz = np.meshgrid(
            np.arange(minb, maxb, step),
            np.arange(minb, maxb, step))

        assert abs((res ** 0.5) ** 2 - res) < 1e-10, \
            'res must be an integer with integer root. Got {}.'.format(res)
        res += 1
        yy = np.zeros(res * res).reshape((res, res))

        # np.stack will create a third axis and stack everything on that
        # so shape will be: rows, cols, 3.
        return np.dstack([xx, yy, zz])

    @staticmethod
    def ray_hit_plane(ray_pt, ray_dir, plane_origin, plane_normal):
        """
        Ray hits plane .
        """
        ray_pt, ray_dir = Point3D.from_array(
            ray_pt), Point3D.from_array(ray_dir)
        pln_n, pln_o = Vector3D.from_array(
            plane_normal), Point3D.from_array(plane_origin)

        # Make geometries
        ray = Ray3D(ray_pt, ray_dir)
        plane = Plane(pln_n, pln_o)

        r = plane.intersect_line_ray(ray)

        if r is None:
            return r

        return np.array(r.to_array())

    @staticmethod
    def ray_hit_plane2(ray_pt, ray_dir, pt_in_plane, plane_normal):
        """
        Toy function. For actual use, use mtx methods.
        R = ray
        AtRx + BtRy + CtRz = d
        t = d/(ARx + BRy + CRz)
        t*R = d/(ARx + BRy + CRz)*R
        """
        # Plane equation
        # d = Ax + By + Cz
        # Use dot for n-dim space
        # norm = A,B,C
        # d != 0 if plane doesn't intersect with origin
        # :return: np.array(A,B,C,...), d

        d = np.dot(plane_normal, pt_in_plane)
        D = np.dot(plane_normal, ray_dir)

        # Check if ray parallel to plane
        if abs(D) < 1e-10:
            return None
        if D > 0.0:
            return None
        #k = np.dot(plane_normal, pt_in_plane)
        #u = (k - np.dot(ray_pt)) / d

        return ray_pt + np.array(ray_dir * d / D)

    @staticmethod
    def ray_hit_polygon(ray_pt, ray_dir, polygon):
        """Return hit point from ray and polygon, if intersection exists
        """

        boundary = [Point3D.from_array(p) for p in polygon]
        face = Face3D(boundary)

        #ipt1 = Pincam.ray_hit_plane2(ray_pt, ray_dir, face.centroid, face.normal)
        ipt = Pincam.ray_hit_plane(ray_pt, ray_dir, face.centroid, face.normal)
        #print(ipt1, ipt)
        if ipt is None:
            return None

        # Multiply all vertices by inverse orthobasis of plane 3d
        poly2d = face.boundary_polygon2d

        # Stack column vectors to make change of basis matrix
        z = np.cross(face.plane.x.to_array(), face.plane.y.to_array())
        basis_mtx = np.array([face.plane.x.to_array(),
                              face.plane.y.to_array(), z]).T
        # Transpose of orthonormal is it's inverse
        ibasis_mtx = basis_mtx.T
        ipt2d = np.matmul(ibasis_mtx, ipt - face.plane.o)
        ipt2d = Point2D(ipt2d[0], ipt2d[1])

        if not poly2d.is_point_inside(ipt2d):
            return None

        return np.array(ipt)

    def depth_buffer(self, ptmtx, default_depths, res=64):
        """Build the depth buffer."""

        default_depths = default_depths[::-1]  # closest geos first

        # Matrix of ray points to compute ray hit
        raymtx = Pincam.ray_hit_matrix(self.sensor_plane_ptmtx_3d, res=res)
        #raymtx: np.dstack([xx, yy, zz]) = (row, col, 3)

        # Build two matrices of geometries for analyis, one 3d, one pixels
        geos = self.view_frustum_geometry2(ptmtx, show_cam=False)

        # Get geos closest to camera
        depth_idx = default_depths[:]
        rownum, colnum, _ = raymtx.shape
        depth_buffer = np.ones(rownum * colnum * 3).reshape(rownum, colnum, 3)
        depth_buffer *= -1
        ray_dir = np.array([0, 1, 0])
        for i in range(rownum):
            for j in range(colnum):
                x, y, z = raymtx[i, j, :]
                ray_pt = np.array([x, y, z])
                for k, geo in enumerate(geos):
                    hitpt = Pincam.ray_hit_polygon(ray_pt, ray_dir, geo)

                    if hitpt is None:
                        continue

                    cur_depth = hitpt[1]  # ydim is depth
                    min_depth = depth_buffer[i, j, 0]
                    cur_geo = k
                    min_geo = depth_buffer[i, j, 1]

                    if min_depth < 0:
                        depth_buffer[i, j, :] = [
                            cur_depth, cur_geo, cur_geo + 2]
                    elif cur_depth < min_depth:
                        depth_buffer[i, j, :] = [cur_depth, cur_geo, 9]
                        depth_idx = Pincam.reorder_depths(
                            depth_idx, cur_geo, min_geo)

        return depth_idx[::-1], depth_buffer

    @staticmethod
    def reorder_depths(depth_idx, cur_geo, min_geo):
        """Reorder depths."""
        min_idx = depth_idx.index(min_geo)
        cur_idx = depth_idx.index(cur_geo)

        if cur_idx < min_idx:
            return depth_idx
        depth_idx.pop(cur_idx)
        depth_idx.insert(min_idx, cur_geo)
        return depth_idx

    @staticmethod
    def project_camera_sensor_geometry(iRt, sensor_plane_3d):
        """Project a reference camera into the scene"""

        cam_srf = np.insert(sensor_plane_3d, 3, 1, 1)
        cam_srf = np.matmul(iRt, cam_srf.T).T
        return cam_srf[:, :3]

    def view_frustum_geometry(self, ptmtx, show_cam=True):
        """View the geometries in the view frustum

        Args:
            ptmtx: List of surfaces as numpy array of points.
            show_ref_cam: Show the camera that is 'viewing' the geometry (Default: True).

        Returns:
            List of surfaces projected in 3d, with reference camera
                as surface.
        """
        # Project sensor, surface geometries in 3d
        _ptmtx = [Pincam.project3d(self.P, pts) for pts in ptmtx]

        # Make into projective points
        _ptmtx = [np.insert(srf, 3, 1, 1) for srf in _ptmtx]

        # Invert the affine transformations (rotation, translation)
        iRt = self.invert_extrinsic_matrix(self.Rt)
        _ptmtx = [np.matmul(iRt, srf.T).T for srf in _ptmtx]
        _ptmtx = [srf[:, :3] for srf in _ptmtx]

        if show_cam:
            camsrf = Pincam.project_camera_sensor_geometry(
                iRt, self.sensor_plane_ptmtx_3d)
            _ptmtx += [camsrf]

        return _ptmtx

    def view_frustum_geometry2(self, ptmtx, show_cam=True):
        """View the geometries in the view frustum w/ geom moved.

        Args:
            ptmtx: List of surfaces as numpy array of points.
            show_ref_cam: Show the camera that is 'viewing' the geometry (Default: True).

        Returns:
            List of surfaces projected in 3d, with reference camera
                as surface.
        """

        _ptmtx = ptmtx
        # Project sensor, surface geometries in 3d
        _ptmtx = [Pincam.project3d(self.P, pts) for pts in _ptmtx]

        if show_cam:
            _ptmtx += [self.sensor_plane_ptmtx_3d]

        return _ptmtx

    def image_matrix(self, ptmtx, inches=10, dpi=10):
        """Construct 2d matrix of geometries.

        Args:
            ptmtx: List of geometries.

        Returns:
            List of image matrices, with dimensions of 120 x 120 x 3
            (row, col, rgb).
        """

        shapes = self.to_gpd_geometry(ptmtx)
        df = gpd.GeoDataFrame({'geometry': shapes})

        # Generate 2d matrices
        for i in range(len(shapes)):
            # Save as png
            fig, ax = plt.subplots(1, figsize=(inches, inches))
            ax.grid(False)
            ax.axis(False)
            ax = df.iloc[i:i + 1].plot(
                edgecolor='black', facecolor='lightblue', lw=5, ax=ax)

            # FIXME: check
            #fig.canvas.draw()
            #img = Image.frombytes(
            #    'RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
            #img = np.array(img)

            fname = 'tmp_{}.png'.format(i)
            plt.savefig(fname, dpi=dpi)
            img = plt.imread(fname)[:,:,:3]

            # Reorder stacks to RGB colors
            b, g, r = np.moveaxis(img, 2, 0)
            img = np.dstack([r, g, b])
            shapes[i] = img

        return shapes

    def to_gpd_geometry(self, ptmtx, res=25):
        """Project geometries to 3d from geopandas dataframe

        Args:
            df: GeoPandas.Dataframe.

        Returns:
            Dataframe with geometry.
        """
        xptmtx, _depths = Pincam.project(self.P, ptmtx)
        depths, _ = self.depth_buffer(ptmtx, _depths, res=res)
        xptmtx = [xptmtx[d] for d in depths]
        return [Pincam.to_poly_sh(srf) for srf in xptmtx]


if __name__ == "__main__":
    # For profiling:
    #  python -m cProfile -s time pincam/pincam.py >> profile.txt
    # 13.105 seconds (raw)
    def r(d): return d / 180.0 * np.pi

    # Define surfaces
    bot_srf = np.array(
        [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]])
    top_srf = np.array(
        [[-5, -5, 10], [5, -5, 10], [5, 5, 10], [-5, 5, 10]])
    y = 0
    vrt_srf = np.array(
        [[-4, y, 0], [4, y, 0], [4, y, 6], [0, y, 10], [-4, y, 6]])
    ptmtx = [bot_srf, top_srf, vrt_srf]

    # Define camera
    focal_length = 20
    heading = r(145)
    pitch = r(0)
    cam_point = np.array([0, -25, 7])
    cam = Pincam(cam_point, heading, pitch, focal_length)

    # Define xptmtx
    res = 64
    xptmtx, _depths = cam.project(cam.P, ptmtx)
    depths, db = cam.depth_buffer(ptmtx, _depths, res=res)
