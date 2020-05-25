from __future__ import print_function, division
from .utils import *
import numpy as np
import shapely.geometry as shapegeom
import shapely.ops as shapeops

class MatrixUtils2(object):
    """
    An improvement of the MatrixUtils static class.
    SparseMatrixUtils works exclusivly with (n x m x p)
    homeogenous, sparse matrix as input and output.

    Impetus for this is from realization there is _never_
    any reason to use a MatrixUtils class on MatrixSurfaces,
    as efficiency for geometry/BEM will come from the
    use of list of matrices.

    Only np no pd.

    Ctrl P in pycharm to see structure.

    """

    @staticmethod
    def matmul_xforms(xforms):
        """Multiply multiply transfromation matrices together.

        This method multiplies matrices in correct post-order multiplication.
        That is matrices are multiplied from left to right.

        For example, for a typical rigid transformation where it is desired to
        multiply rotation around Z axis, rotation around X, and translation, the
        method does the following:

        xforms: [RZ1, RX2, T1]
        P = T1(RX2(RZ1))

        Args:
            list_of_matrices: list of matrices in sequential order.

        Returns:
            Single matrix.
        """
        m = xforms[0]
        for xf in xforms[1:]:
            m = np.matmul(xf, m)
        return m


    @classmethod
    def affine_xform(cls, xform, vertsmtx, w=1):
        """S
        Ref: http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/
        To apply a 4x4  transformation we need to make
         stacked matrix 4x4, and then transpose

        Affine transformation is a linear mapping method that
        preserves points, straight lines, and planes.
        A set of parallel lines will remain paralle
        after an affine transfromation (reflection, rotation, projection).

        verts_mtz = [[x,y,z],
                     [x,y,z],
                     [x,y,z]]

        :param vertsmtz: (n x m x p) sparse matrix
        :param xform: sequence or individual transformation matrices.
        :return: stacked matrix (NOT sparse matrix!)

        """
        # obj= index (before) to add, values=what to add
        # This makes world coords into homogeonous coordinates
        vertsmtx4d = np.insert(vertsmtx, obj=3, values=w, axis=1)  # add a column on col num 3
        # vertsmtx = [X, Y, Z, W], where W = homogenous coordinates 1
        if type(xform) is list or type(xform) is tuple:
            xform = cls.multiply_mtx_seq(xform)
        #pp(xform)
        #pp(vertsmtx)
        xformed_mtx = np.matmul(xform, vertsmtx4d.T).T
        xformed_mtx = xformed_mtx[:,:3]
        #pp(xformed_mtx)

        return xformed_mtx

    @classmethod
    def apply_xform(cls, sparse_matrix, xform, w=1):
        """
        Rotates sparse_matrix based on sequence of xforms.
        np.einsum might achieve this in a vectorized fashion.
        sparse_matrix = [[[x,y,z],
                          [x,y,z],
                          [x,y,z]]]
        :param sparse_matrix: (n x m x p) sparse matrix
        :param xform: sequence of transformation matrices.
        :return: sparse matrix
        """
        # Curry this for efficient map
        def curry_xform(_xform):
            """
            Curried function for apply multiple xforms
            to multiple arrays in a sequence.
            """

            def g(_sparse_arr):
                a = _sparse_arr
                for xf in _xform:
                    a = np.matmul(xf, a)
                return a
            return g

        seq_num = sparse_matrix.shape[0]
        row_num = sparse_matrix.shape[1]

        # To apply a 4x4  transformation we need to make
        # sparse_matrix 4x4, and then transpose
        # obj= index (before) to add, values=what to add
        # This makes world coords into homogeonous coordinates
        sparse_matrix = np.insert(sparse_matrix, obj=3, values=w, axis=2)  # add a column on col num 3
        # sparse_matrix = [X, Y, Z, W], where W = homogenous coordinates 1
        sparse_matrix = sparse_matrix.transpose(0, 2, 1)        # transpose 1,2 axis

        xform_lambda = curry_xform(xform)
        #pp('chk')
        #pp(sparse_matrix)
        xformed_seq = map(xform_lambda, sparse_matrix)
        xformed_matrix = np.vstack(xformed_seq).reshape(seq_num, 4, row_num)
        xformed_matrix = xformed_matrix.transpose(0, 2, 1)    # swap it back
        xformed_matrix = np.delete(xformed_matrix, 3, axis=2)   # from 4

        return xformed_matrix

    @classmethod
    def multiply_mtx_seq(cls, xform_seq):
        """
        Multiples sequence of matrices to produce one matrix.
        :param matrix_seq:
        :return:matrix
        """
        ident = np.identity(4)
        for xf in xform_seq:
            ident = np.matmul(ident, xf)
        return ident

    @classmethod
    def sparse_matrix_from_seq(cls, seq):
        """
        Generates sparse matrix from sequence of ndarrays.
        :param seq: nd.array or list of unequal nd.arrays
        :return: sparse matrix
        """

        def helper_pad_array(arr, sparse_row_len):
            """
            Pads input array with np.nan rows
            to makes array sparse
            :param arr: array
            :param sparse_row_len: length of sparse rows to add
            :return: array
            """
            rows_to_add = sparse_row_len - arr.shape[0]

            if rows_to_add > 0:
                nanrows = np.empty(3 * rows_to_add)
                nanrows[:] = np.nan
                nanrows = nanrows.reshape(rows_to_add, 3)
                sparse_arr = np.vstack((arr, nanrows))
            else:
                sparse_arr = arr

            return sparse_arr

        arrlen = [x.shape[0] for x in seq] # number of rows
        maxlen = arrlen[np.argmax(arrlen, axis=0)]
        padded_seq = [helper_pad_array(v, maxlen) for v in seq]
        sparse_matrix = np.vstack(padded_seq).reshape(len(seq), maxlen, 3) # reshape in ndarray
        return sparse_matrix

    @classmethod
    def stacked_matrix_from_seq(cls, array_lst, axis=0):
        """
        Ref: https://tonysyu.github.io/ragged-arrays.html#.XLwA8-hKiHs
        Consumes list of arrays of different lenghts.
        Returns:
            stacked: flattened array of points
            idx: list of indicees of arrays. Use np.split(stack, array)
            to retrieve former array structure.
        """
        # Get the length of array
        len_lst = [np.shape(a)[axis] for a in array_lst]
        # List of cumulative sum of array lenghts, except last one.
        # idx = index, list of indices for every array
        idx = np.cumsum(len_lst[:-1])
        # Now flatten array_lst
        stacked = np.concatenate(array_lst, axis=axis)

        return stacked, idx

    @classmethod
    def xform_plane_project_matrix(cls, proj_dir_vec, plane_norm, plane_d):
        """
        Warning: Does not care if proj_dir_vec in wrong direction, will just reverse projection matrix.

        Ref: https://stackoverflow.com/questions/2500499/howto-project-a-planar-polygon-on-a-plane-in-3d-space

        Derivation:
        (x,y,z) = (x0 + tdx, y0 + tdy, z0 + tdz), where t is unknown, and (dx,dy,dz) is dirn vector
        Derive t by plugging it into the plane equation:
        d = Ax + By + Cz
        d = A(x0 + tdx) + B(y0 + tdy) + C(z0 + tdz)
          = Ax0 + Atdx + By0 + Btdy + Cz0 + Ctdz
          = t * (Adx + Bdy + Cdz) + Ax0 + By0 + Cz0
        t = (d - Ax0 - By0 - Cz0)/(Adx + Bdy + Cdz)

        set D as (Adx + Bdy + Cdz), or np.dot(plane_norm, dirn_vec)
        x = x0 + tdx
          = x0 + d*dx/D - A*x0*dx/D - B*y0*dx/D - C*z0*dx/D
          = x0 -A*x0*dx/D -B*y0*dx/D -C*z0*dx/D +d*dx/D

        And for y,z:
        y = -A*x0*dx/D +y0-B*y0*dx/D -C*z0*dx/D d*dx/D
        z = -A*x0*dx/D -B*y0*dx/D +z0-C*z0*dx/D d*dx/D
        In matrix form:
        xf = [[1-A*dx/D,  -B*dx/D,  -C*dx/D, d*dx/D],
              [ -A*dy/D, 1-B*dy/D,  -C*dy/D, d*dy/D],
              [ -A*dz/D,  -B*dz/D, 1-C*dz/D, d*dz/D],
              [       0,        0,        0,       1]]
        """
        # Check if plane is collinear w/ dir_vec
        D = np.dot(proj_dir_vec, plane_norm)
        if is_near_zero(D, eps=1e-10):
            return None

        # t param tells us direction of the project
        # We ignore this.
        # t =d/D

        A,B,C = plane_norm
        d = plane_d
        dx,dy,dz = proj_dir_vec

        xf = [[1-A*dx/D,  -B*dx/D,  -C*dx/D, d*dx/D],
              [ -A*dy/D, 1-B*dy/D,  -C*dy/D, d*dy/D],
              [ -A*dz/D,  -B*dz/D, 1-C*dz/D, d*dz/D],
              [       0,        0,        0,      1]]
        xf = np.array(xf)

        return xf

    @classmethod
    def xform_translation_matrix(cls, move_vector):
        """
        Modified from Christopher Gohlke
        https://www.lfd.uci.edu/~gohlke/code/transformations.py.html

        :param move_vector: 3 x 1 direction vector
        :return: 4 x 4 translation matrix
        [[1, 0, 0, move_vector[0]],
        [0, 1, 0, move_vector[1]],
        [0, 0, 1, move_vector[2]],
        [0, 0, 0, 1.0]]
        """
        m = np.identity(4)
        m[:3, 3] = move_vector[:3]
        return m

    @classmethod
    def xform_rotation_matrix(cls, vector_origin, vector_axis, theta):
        """
        Modified from Christopher Gohlke
        https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
        Return ndarray normalized by length, i.e. Euclidean norm, along axis.

        If this doesn't work, go back to Rodriguez rotation method in matrixutils.
        """

        sina = np.sin(theta)
        cosa = np.cos(theta)
        vector_axis = vector_axis/np.linalg.norm(vector_axis)

        # rotation matrix around unit vector
        R = np.diag([cosa, cosa, cosa])
        R += np.outer(vector_axis, vector_axis) * (1.0 - cosa)
        vector_axis *= sina
        R += np.array([[ 0.0,         -vector_axis[2],  vector_axis[1]],
                       [ vector_axis[2], 0.0,          -vector_axis[0]],
                       [-vector_axis[1], vector_axis[0],  0.0]])
        M = np.identity(4)
        M[:3, :3] = R

        if vector_origin is not None:
            # rotation not around origin
            M[:3, 3] = vector_origin - np.dot(R, vector_origin)

        return M

    @classmethod
    def xform_reflection_matrix(cls, plane_point, plane_normal):
        """
        Modified from Christopher Gohlke
        Return matrix to mirror at plane defined by point and normal vector.
        """
        plane_normal = plane_normal/np.linalg.norm(plane_normal[:3])
        M = np.identity(4)

        # Compute outer product of two vectors
        M[:3, :3] -= 2.0 * np.outer(plane_normal, plane_normal)
        M[:3, 3] = (2.0 * np.dot(plane_point[:3], plane_normal)) * plane_normal

        return M

    @classmethod
    def xform_scale_matrix(cls, factor, origin=None, direction=None):
        """
        Modified from Christopher Gohlke
        Return matrix to scale by factor around origin in direction.
        Use factor -1 for point symmetry.

        >>> v = (numpy.random.rand(4, 5) - 0.5) * 20
        >>> v[3] = 1
        >>> S = scale_matrix(-1.234)
        >>> numpy.allclose(numpy.dot(S, v)[:3], -1.234*v[:3])
        True
        >>> factor = random.random() * 10 - 5
        >>> origin = numpy.random.random(3) - 0.5
        >>> direct = numpy.random.random(3) - 0.5
        >>> S = scale_matrix(factor, origin)
        >>> S = scale_matrix(factor, origin, direct)

        """
        if direction is None:
            # uniform scaling
            M = np.diag([factor, factor, factor, 1.0])
            if origin is not None:
                M[:3, 3] = origin[:3]
                M[:3, 3] *= 1.0 - factor
        else:
            # nonuniform scaling
            direction = unit_vector(direction[:3])
            factor = 1.0 - factor
            M = np.identity(4)
            M[:3, :3] -= factor * np.outer(direction, direction)
            if origin is not None:
                M[:3, 3] = (factor * np.dot(origin[:3], direction)) * direction
        return M


    @classmethod
    def xform_change_basis_plane_matrix(cls, ref_ortho_basis, pt_in_plane):
        """
        Change of basis
        Project onto old x/y basis by taking dot btwn y/x and Point
        Resulting value = vector length of new x/y basis, repeat for y and add

        Ref: https://stackoverflow.com/questions/44553886/how-to-create-2d-plot-of-arbitrary-coplanar-3d-curve/44559920
        https://stackoverflow.com/questions/23814234/convert-3d-plane-to-2d

        Intuition: https://www.khanacademy.org/math/linear-algebra/alternate-bases/change-of-basis/v/linear-algebra-coordinates-with-respect-to-a-basis

        CCW orientation is preserved in ortho_basis
        """
        m = np.identity(4)

        # This creates matrix of dot products of old vectors and new vectors
        # Usually we transpose to ensure points (in rows) are switched to cols
        # but am unsure if this is the case here. For now transpose.
        xform_mtx = ref_ortho_basis
        m[:3,:3] = xform_mtx

        # Reminder, don't transpose xforms
        tmtx = MatrixUtils2.xform_translation_matrix(pt_in_plane * -1)
        m = np.matmul(m, tmtx)

        return m

    @classmethod
    def shapelysrf_contains_point(cls, shapely_srf, shapely_pt):
        # Toy fx
        return shapely_srf.contains(shapely_pt)

    @classmethod
    def normalize(indfmtx, lo=0.0, hi=1.0):
        """
        Normalize matrix or pandas dataframe.
        If matrix coming from pandas, make sure to transpose as
        pandas.as_matrix() nests row.
        :param inmtx: dataframe or matrix of same dtype

        """
        maxv = np.nanmax(indfmtx)
        minv = np.nanmin(indfmtx)

        # Normalize
        ndfmtx = (hi - lo) * (indfmtx - minv)/(maxv - minv)

        return ndfmtx

    @classmethod
    def plane_equation(cls, normal, pt_in_plane):
        """
        d = Ax + By + Cz
        Use dot for n-dim space
        norm = A,B,C
        d != 0 if plane doesn't intersect with origin
        :return: np.array(A,B,C,...), d
        """
        d = np.dot(normal, pt_in_plane)
        return normal, d

    @classmethod
    def ray_to_ray_intersect_2d(cls, r1_pt, r1_dir, r2_pt, r2_dir):
        """
        Determine if there's an intersection between two rays
        Theory: Computational Geometry in C, Chapter 1, Section 1:
        ode ref: https://stackoverflow.com/questions/2931573/determining-if-two-rays-intersect

        Finds u and v scalars for both rays

        :param ray1: (vo + v*v1)
        :param ray2: (vo + u*v1)
        :return: pt
        """
        dx = r1_pt[0] - r2_pt[0]
        dy = r2_pt[1] - r2_pt[1]

        # Determinate of direction mtx [r1dir, r2dir]^T
        det = r1_dir[0] * r2_dir[1] - r2_dir[0] * r1_dir[1]

        if is_near_zero(det, 1e-10):
            return None

        u = (dy * r2_dir[0] - dx * r2_dir[1]) / det

        # Skip this since we only want the end point, not scalars
        #v = (dy * r1_dir[0] - dx * r1_dir[1]) / det

        return r2_pt + u * r2_dir

    @classmethod
    def project_ray_to_plane(cls, ray, plane_normal, d):
        """
        Toy function. For actual use, use mtx methods.
        R = ray
        AtRx + BtRy + CtRz = d
        t = d/(ARx + BRy + CRz)
        t*R = d/(ARx + BRy + CRz)*R
        """
        D = np.dot(plane_normal, ray)

        # Check if ray parallel to plane
        if is_near_zero(D):
            return None

        return ray * d/D


    @classmethod
    def xy_plane_ortho_basis(cls):
        """
        Returns matrix representing a set B of orthogonal elements (vectors) for XY plane

        :return: nd.array(x, y, z)
        """
        return np.array(
            [[1,0,0],
             [0,1,0],
             [0,0,1]]
        )

    @classmethod
    def cam_proj(cls, sqarr, view_extents, dxy, dz, stacked=True):
        """
        Reg: http://docs.enthought.com/mayavi/mayavi/auto/example_mlab_3D_to_2D.html
        Camera projection
        Transform stacked matrix of coordinates of all surfaces to camera projection
        # dxy rotates xy-plane clockwise
        # dz rotates z-axis towards screen

        # transform = R * T
        # P = Projection * View
        # final = transform x P

        """
        def helper_view_to_disp_matrix(xlen,ylen):
            # Curried.
            return np.array(
                [[xlen/1.0,         0.,   0.,   xlen/1.0],
                 [      0.,  -ylen/1.0,   0.,   ylen/1.0],
                 [      0.,         0.,   1.,         0.],
                 [      0.,         0.,   0.,         1.]]
            )

        def helper_cam_xforms(rotation_angle_y, rotation_angle_z):
            """
            Projection plane rotation.
            :param rotation_angle_y:
            :param rotation_angle_z:
            :return:
            TODO: I think this is better:
            https://stackoverflow.com/questions/23472048/projecting-3d-points-to-2d-plane
            """
            return [
                MatrixUtils2.xform_rotation_matrix(
                    np.array([0,0,0]),
                    np.array([0,0,1]),
                    rotation_angle_y
                ),
                MatrixUtils2.xform_rotation_matrix(
                    np.array([0,0,0]),
                    np.array([1,0,0]),
                    rotation_angle_z
                ),
                # TODO: Hack fix with rewrite
                MatrixUtils2.xform_reflection_matrix(
                    np.array([0,0,0]),
                    np.array([0,1,1])
                )
            ]

        if stacked: sqarr = np.expand_dims(sqarr, axis=0)

        # Generate camera matrices
        cam_xforms = helper_cam_xforms(np.radians(dxy), np.radians(dz))
        view_to_disp_matrix = helper_view_to_disp_matrix(*view_extents)
        cam_xforms = cam_xforms + [view_to_disp_matrix]
        # TODO: rotate all points then find distance to projected plane
        # (sum of squares from 3d pt to plane)
        # find np.max for eeach sqarrr row
        # that is our depth buffer
        #proj_sqarr = MatrixUtils2.affine_xform(sqarr, cam_xforms)
        proj_sqarr = MatrixUtils2.apply_xform(sqarr, cam_xforms)

        if stacked: proj_sqarr = proj_sqarr[0]

        geom_lst = np.delete(proj_sqarr,2,axis=1)
        depthbuff_lst = proj_sqarr[:,2]

        return geom_lst, depthbuff_lst

    @classmethod
    def edge_cardinal_orientation(cls, dirvec):
        # Assume edge is CCW
        """
        Return orientation in degrees
        """
        if np.abs(dirvec[0]) > np.abs(dirvec[1]): # edge is running EW
            if(not is_near_zero(dirvec[0]) and dirvec[0] > 0.0): # edge is running E and facing S
                return 180    # S Wall is running E and facing S
            else:
                return 0      # N Wall is running W and facing N
        else:
            if not is_near_zero(dirvec[1]) and dirvec[1] > 0.0:
                return 90     # E Wall is running N and facing E
            else:
                return 270    # W Wall is running S and facing W

    @classmethod
    def vector_cardinal_orientation(cls, dirvec):
        """
        Return orientation in degrees
        """
        # Vector is judged horizontal orE/W when x>y, vertical or N/S when y>x
        if np.abs(dirvec[0]) > np.abs(dirvec[1]): # vec is pointing E/W
            if(not is_near_zero(dirvec[0]) and dirvec[0] > 0.0): #
                return 90 # E
            else:
                return 270 # W
        else:
            if not is_near_zero(dirvec[1]) and dirvec[1] > 0.0:
                return 0 # N
            else:
                return 180 # S

    @classmethod
    def groupby(cls, ndarr, tol=1e-10, axis_lambda=None):
        """
        Group arrays according to tol.
        Axis_lambda is used to identify axis you are grouping by (default is z coord of srfarr).
        Sorts stably.
        Preserves order so we should be able to extract edges from this.
        Best used on 2d planar geometries with simplified curves though.
        """
        if axis_lambda is None:
            axis_lambda = lambda nx: nx[:,2]

        # Stable sort ccw direction of edges remain
        ndarr = ndarr[axis_lambda(ndarr).argsort(kind='mergesort')]
        zaxis = axis_lambda(ndarr)

        # Sort zaxis, and return mask of where diff exceed tol
        bool_mask_diff = np.diff(zaxis) > tol

        # Get list of indices where the change occurs
        diff_indices = np.where(bool_mask_diff == True)[0] + 1

        # Group
        grouped_ndarr = np.split(ndarr, diff_indices)

        return grouped_ndarr, diff_indices

    @classmethod
    def extrude_points(cls, pts_arr, norm):
        """
        Extrude array of pts into array of lines
        """
        # Make top vector by adding pts_arr with norm
        top_arr = pts_arr + norm
        extr_pts_arr = np.array(list(zip(pts_arr,top_arr)))
        return extr_pts_arr

    @classmethod
    def extrude_line(cls, line, norm):
        """
        line is array of two vertices.
        Extrude based on normal
        """
        v0, v1 = line[0], line[1]
        # Make new vertices
        v2 = v1 + norm
        v3 = v2 + (v0 - v1)
        extr = np.array([v0, v1, v2, v3], dtype=np.float64)
        #pp(extr)
        return extr

    @classmethod
    def extrude_srf(cls, base_srf, edgemtx, dirvec):
        """Extrude srf"""
        extruded_srfs = [cls.extrude_line(edge, dirvec) for edge in edgemtx]
        xf = cls.xform_translation_matrix(dirvec)
        cap_srf = cls.affine_xform(xf, base_srf)

        return [base_srf, *extruded_srfs, cap_srf]

    @classmethod
    def is_parallel(cls, v1, v2):
        d = np.dot(v1/np.linalg.norm(v1), v2/np.linalg.norm(v2))
        return is_near_zero(1.0 - d)

    @classmethod
    def srf_normal(cls, edges_dir):
        """
        Will traverse through edges of srf until finds non-colinear edges
        if neccessary.
        Unitized
        """
        edge_idx = 0
        is_par = cls.is_parallel(edges_dir[edge_idx], edges_dir[edge_idx+1])


        # Find non-colinear edges
        while is_par == True and edge_idx < len(edges_dir):
            edge_idx += 1
            is_par = cls.is_parallel(edges_dir[edge_idx], edges_dir[edge_idx+1])

        norm = np.cross(edges_dir[edge_idx], edges_dir[edge_idx+1])
        unorm = norm/np.linalg.norm(norm)
        return unorm

    @classmethod
    def srf_edges(cls, srf_vert_loop):
        """
        Return edges as 3d matrix
        :param srf:
        :return: [[vertice0, vertice1],
                  [vertice1, vertices2].
                  ....]
        """
        srf = srf_vert_loop
        edgemtx = [[srf[i], srf[i+1]] for i in range(len(srf) - 1)]
        edgemtx = np.array(edgemtx)
        return edgemtx

    @classmethod
    def srf_edges_dir(cls, srf_edges):
        """ Not Unit vectors
        """
        return srf_edges[:,1,:] - srf_edges[:,0,:]

    @classmethod
    def unitize_vectors(cls, dir_arr, dist_arr):
        # Broadcast rules divide scalalr dist for dir vectors along axis 0
        # Toy fx
        # ref for div: https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element

        return (dir_arr.T/dist_arr).T

    @classmethod
    def srf_edges_len(cls, edge_vec):
        return np.linalg.norm(edge_vec, axis=1)

    @classmethod
    def shapely_from_srf3d(cls, srf):
        """ Srf cab be 3dm shapely init will automatically remove extra dims"""
        return shapegeom.Polygon(srf[:,:2])

    @classmethod
    def shapely_from_point3d(cls, ptarr):
        """Array of 3 coordinates
        """
        return shapegeom.Point(ptarr[:2])

    @classmethod
    def shapely_from_line3d(cls, linearr):
        ptsarr = [cls.shapely_from_point3d(v) for v in linearr]
        return shapegeom.LineString(ptsarr)

    @classmethod
    def shapely_to_point3d(cls, sgpt, w=None):
        """
        Takes a shapely point and returns 3d point array
        :param sgpt: shapely point
        :return: np.ndarry[x,y,z]
        """
        pt = np.hstack(sgpt.coords.xy)
        if w:
            pt = np.insert(pt, values=[0,w], obj=2, axis=0)
        else:
            pt = np.insert(pt, values=0, obj=2, axis=0)

        return pt

    @classmethod
    def shapely_to_line3d(cls, shape):
        return np.vstack(shape.coords)

    @classmethod
    def shapely_to_srf3d(cls, shape):
        mtx2d = np.vstack(shape.exterior.coords)
        return np.insert(mtx2d, values=0, obj=2, axis=1)

    @classmethod
    def shapely_from_shapemtx3d(cls, mtx):
        """
        Handles polygons, lines, and points
        :param mtx: matrix of vertice(s)
        :return: shapely object
        """
        if mtx.shape[0] > 2:
            shape = cls.shapely_from_srf3d(mtx)
        elif mtx.shape[0] > 1:
            shape = cls.shapely_from_line3d(mtx)
        else:
            shape = cls.shapely_from_point3d(mtx[0])

        return shape

    @classmethod
    def shapely_to_shapemtx3d(cls, shape):
        """
        Interprets if the shape is a line, polygon, point
        and returns appropriate numpy matrix of vertices
        :param shape: shapely geom
        :return: nd.array of vertices. Even point will be wrapped in a list.
        """
        if hasattr(shape, 'exterior'):
            vertmtx = np.vstack(shape.exterior.coords)
        else:
            vertmtx = np.vstack(shape.coords)

        return np.insert(vertmtx, values=0, obj=2, axis=1)

    @classmethod
    def shapely_centroid(cls, shape):
        return shape.centroid.coords[0]

    @classmethod
    def shapely_approx_centroid(cls, shape):
        """ Computationally chepaer then shapely_centroid"""
        return shape.representative_point().coords[0]

    @classmethod
    def shapely_convexhull(cls, shape):
        return shape.convex_hull

    @classmethod
    def shapely_is_empty(cls, shape):
        # toy fx
        return shape.is_empty

    @classmethod
    def ortho_basis_from_normal(cls, vec_norm, vec_to_right):
        """
        Take normal and vector in plane to identify
        the x and y basis vectors and return x,y,z
        as column vectors.

        We want vec_to_right to in x axis pointing to right
        for our u,v convention.

        :param vec_norm: vector to orthogonalize
        :param ortho_basis_mtx: [xvec, yvec, zvec]]
        :return:
        """

        y_basis = np.cross(vec_norm, vec_to_right)
        x_basis = np.cross(y_basis, vec_norm)

        y_basis = y_basis/np.linalg.norm(y_basis)
        x_basis = x_basis/np.linalg.norm(x_basis)

        return np.array([x_basis, y_basis, vec_norm])

    @classmethod
    def map_shapely_to_srf(
            cls,
            shape_fx,
            xform_shapely_basis,
            xform_shapely_inv_basis,
            *shape_args
    ):
        """
        Function takes a shapely geom, applies
        a shapely fx and then maps result back to 3d basis
        :xform_shapely_basis: ortho basis for 2d shapely object (xy plane)
        :xform_shapely_inv_basis: inverse transformation matrix back to 3d basis
        :shape_fx: function for shapely method
        :shape_args: args for shapely method if any
        :return: shapely output
        """
        # Transform our srf to shapely ortho basis with the xform
        shape_args = (
            cls.affine_xform(xform_shapely_basis, arg) for arg in shape_args
        )
        shape_args = (
            cls.shapely_from_shapemtx3d(arg) for arg in shape_args
        )
        result = shape_fx(*shape_args)

        # If result is error, remember:
        # We use MatrixSurface2(vertmtx).shapely if result is primitive
        # This fx should only be used for geometry outputs!
        # See fx_point_in_srf for example

        # Convert to np and project back to our original srf basis
        vertmtx = cls.shapely_to_shapemtx3d(result)

        #vertmtx = cls.affine_xform(xform_shapely_inv_basis, vertmtx)
        vertmtx4d = np.insert(vertmtx, values=1, obj=3, axis=1)
        ixf = xform_shapely_inv_basis
        vertmtx = np.matmul(ixf, vertmtx4d.T).T
        vertmtx = vertmtx[:,:3]

        return vertmtx

    @classmethod
    def is_srf_projectable(cls, src_msrf, tgt_msrf):

        tsrf = tgt_msrf
        ssrf = src_msrf

        tgt_norm, _ = tsrf.plane_eqn
        src_norm, src_d = ssrf.plane_eqn

        # Check if dirn vector is parallel to plane
        D = np.dot(src_norm, tgt_norm)
        if is_near_zero(D):
            return False

        # Check if t factor for dirn vector (-tgt_norm) might reverse factor
        t = src_d/D

        #if t < 0.0:
        #    return False

        # Check if src cpt can be projected into target srf
        # by projecting src cpt onto the plane, then checking if
        # it lands in the polygon bounds
        proj_xf = cls.xform_plane_project_matrix(tgt_norm*-1, *tsrf.plane_eqn)
        test_pt = cls.affine_xform(proj_xf, np.array([ssrf.cpt]))[0]

        is_in = tsrf.fx_point_in_srf(test_pt)

        return is_in

    @classmethod
    def project_msrf_to_msrf(cls, src_msrf, tgt_msrf):
        """ Returns projected src_msrf to tgt_msrf.
        If not possible to project, returns None.
        """
        # Check if dirn matches
        is_proj = cls.is_srf_projectable(src_msrf, tgt_msrf)

        if not is_proj:
            return None

        proj_xf = cls.xform_plane_project_matrix(tgt_msrf.normal*-1, *tgt_msrf.plane_eqn)
        proj_vertmtx = cls.affine_xform(proj_xf, src_msrf.vertmtx)

        return proj_vertmtx

    @classmethod
    def shapely_union(cls, shapely_srf_seq):
        """
        Consumes collection of 2d vertmtx (no 3rd dim) and finds boolean union.
        :param vertmtx_2d_seq:
        :return:booleaned shapely surf
        """
        # toy fx
        return shapeops.cascaded_union(shapely_srf_seq)

    @classmethod
    def squircle(cls, p=2.5):
        """
        For squircles (i.e apple rounded corners).
        This just plots an example, run in jupyter.
        For real application, return the x,y arrays.

        Ref: https://www.johndcook.com/blog/2018/02/13/squircle-curvature/
        2 = circle
        3 = squircles
        > 4 square

        """
        def _rad(p,t):
            d = np.power(np.abs(np.cos(t)), p) +\
                np.power(np.abs(np.sin(t)), p)
            return 1/np.power(d,(1/p))

        # Polar coordinates
        theta = np.arange(360,dtype=np.float64)*np.pi/180.
        rad = _rad(p,theta)

        # Convert to cartesion
        x = np.cos(theta)/rad
        y = np.sin(theta)/rad

        # Rotate by 45
        pts = np.vstack([x,y])
        t = np.pi/4. # viz theta
        rmtx = np.array([[np.cos(t), -np.sin(t)],
                         [np.sin(t),  np.cos(t)]])
        rmtx = rmtx.T # sinces every arrays = columns
        pts = np.matmul(rmtx, pts) # confusingly, pts remains non columns

        # Plot
        f,a = plt.subplots(1,1)
        a.plot(pts[0,:], pts[1,:])
        a.axis('equal')
