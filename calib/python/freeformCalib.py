import argparse
import os
import sys

import autograd
# import tiffcapture as tc #using jpg for now
import autograd.numpy as np
import cv2
import scipy
import scipy.integrate
import scipy.optimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'python/lib/'))
import logger

if not __name__ == "__main__":
    import traceback

    filename = traceback.format_stack()[0]
    log = logger.getLogger(filename.split('"')[1], False, False)


def singleCamCalib(umeas, xworld, planeData, cameraData):
    """
    Inputs
    -------------------------------------------
    umeas              2 x nX*nY*nplanes x ncams array of image points in each camera
    planeData          struct of image related parameters
    cameraData         struct of camera related parameters

    Returns
    -------------------------------------------
    cameraMats         3 x 3 x ncams: camera matrices for individual cameras
    boardRotMats       3 x 3*nplanes x ncams: board rotation matrix per image per camera
    boardTransVecs     3 x nplanes x ncams: board translational vector per image per camera
    """

    log.info("Single camera calibrations")

    # set up implementation variables
    nX = planeData.nX
    nY = planeData.nY
    nplanes = planeData.ncalplanes
    ncams = cameraData.ncams
    imageSize = (cameraData.sX, cameraData.sY)
    aspectRatio = cameraData.sX / cameraData.sY

    # set up storage variables
    cameraMats = np.zeros([3, 3, ncams])
    boardRotMats = np.zeros([3, 3, nplanes, ncams])
    boardTransVecs = np.zeros([3, nplanes, ncams])

    # cycle through each camera
    log.info("Cycle through each camera")
    for c in range(0, ncams):
        log.VLOG(2, "Calibrating camera %d", c)

        # calculate initial camera matrix for this camera
        # use all world points but only image points in this camera
        camUmeas = np.transpose(umeas[:, :, c]).astype('float32')

        camMatrix = cv2.initCameraMatrix2D([xworld], [camUmeas], imageSize, aspectRatio)
        log.VLOG(4, "with Intrinsic Parameter Matrix %s", camMatrix)

        # iterate through each board placement to find rotation matrix and rotation vector
        for n in range(0, nplanes):
            log.VLOG(3, "%%% on image {}".format(n))
            # isolate only points related to this board's placement
            currWorld = xworld[nX * nY * n: nX * nY * (n + 1), :]
            currImage = camUmeas[nX * nY * n: nX * nY * (n + 1), :]

            # find the board's rotation matrix and translation vector

            _, rotMatrix, transVector = cv2.solvePnP(currWorld, currImage, camMatrix,
                                                     np.zeros((8, 1), dtype='float32'))
            log.VLOG(4, "Rotation matrix for camera %d image %d: %s", c, n, rotMatrix)
            log.VLOG(4, "Translational vector for camera %d image %d: %s", c, n, transVector)

            # add board values to storage variables
            boardRotMats[:, :, n, c] = rotMatrix
            boardTransVecs[:, n, c] = transVector.ravel()

        # add camera matrix to storage variable
        cameraMats[:, :, c] = camMatrix

    return [cameraMats, boardRotMats, boardTransVecs]


def kabsch(X1, X2):
    """Returns pairwise rotation matrix and translation vector by
       computing the cross-covariance matrix given two set of world points.

       Inputs
       ------------------------
        # X1: The world coordinates of the first camera stacked as columns.
        # X2: The world coordinates of the second camera stacked as columns.
    """

    x1m = X1.mean(axis=1)
    x2m = X2.mean(axis=1)

    # Subtract mean positions.
    X1_centered = np.zeros(X1.shape)
    X2_centered = np.zeros(X2.shape)

    for i in range(X1.shape[0]):
        X1_centered[i, :] = X1[i, :] - x1m[i]
        X2_centered[i, :] = X2[i, :] - x2m[i]

    A = np.matmul(X2_centered, np.transpose(X1_centered))

    U, S, Vtrans = np.linalg.svd(A)

    # calculates variable to correct rotation matrix if necessary
    d = np.linalg.det(np.matmul(U, Vtrans))
    D = np.eye(3)
    D[2, 2] = d

    R_12 = np.matmul(np.matmul(U, D), Vtrans)
    t_12 = np.mean(X2 - np.matmul(R_12, X1), 1)

    return R_12, t_12


def quaternionToRotMat(q):
    """Convert 1 x 4 quaternion to a 3 x 3 rotation matrix.
       (Unclear what quaternions correspond to)
    """

    normalizationFactor = np.dot(q, q)

    # qc is a row vector
    qc = np.array([[q[1], q[2], q[3]]])
    # Cross operator matrix.
    Q = np.array([[0, -q[3], q[2]], [q[3], 0, -q[1]], [-q[2], q[1], 0]])

    R = (q[0] ** 2 - np.dot(qc, qc)) * np.eye(3) + 2 * np.matmul(np.transpose(qc), qc) + 2 * q[0] * Q;
    R = R / normalizationFactor
    return R


def rotMatToQuaternion(R):
    """
    Convert 3x3 rotation matrix to 1x4 quaternion
    :param R: 3x3 rotation matrix
    :return: quaternion matrix
    """
    # component-wise from rotation formalism
    qr = 1 / 2 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    qi = (R[2, 1] - R[1, 2]) / (4 * qr)
    qj = (R[0, 2] - R[2, 0]) / (4 * qr)
    qk = (R[1, 0] - R[0, 1]) / (4 * qr)

    return [qr, qi, qj, qk]


def multiCamCalib(umeas, xworld, cameraMats, boardRotMats, boardTransVecs, planeData, cameraData):
    """Compute the calibrated camera matrix, rotation matrix, and translation vector for each camera.

    Inputs
    -------------------------------------------
    umeas              2 x nX*nY*nplanes x ncams array of image points in each camera
    xworld             3 x nX*nY*nplanes of all world points
    cameraMats         3 x 3 x ncams that holds all camera calibration matrices; first estimated in single camera calibration
    boardRotMats       3 x 3*nplanes x ncams for each board in each camera; from single camera calibration.
    boardTransVecs     3 x nplanes x ncams for each board in each camera; from single camera calibration
    planeData          struct of image related parameters
    cameraData         struct of camera related parameters

    Returns
    -------------------------------------------
    cameraMats         3 x 3 x ncams; final estimate of the camera matrices
    rotationMatsCam    3 x 3 x ncams; final estimate of the camera rotation matrices
    transVecsCam       3 x ncams; final estimate of the camera translational vectors
    """

    # loop over all camera pairs
    # call kabsch on each pair
    # store rotation matrices and translational vectors
    # Calculate terms in sum, add to running sum

    log.info("Multiple Camera Calibrations")

    # Extract dimensional data.
    ncams = cameraData.ncams

    # Storage variables.
    R_pair = np.zeros([3, 3, ncams, ncams])
    t_pair = np.zeros([3, ncams, ncams])

    for i in range(0, ncams):
        for j in range(i + 1, ncams):
            log.VLOG(2, 'Correlating cameras (%d, %d)' % (i, j))

            # Compute world coordinates used by kabsch.
            Ri = boardRotMats[:, :, :, i].ravel().reshape((1, -1, 3))[0]
            ti = boardTransVecs[:, :, i].reshape((-1, 1))
            Rj = boardRotMats[:, :, :, j].ravel().reshape((1, -1, 3))[0]
            tj = boardTransVecs[:, :, j].reshape((-1, 1))

            # Compute world coordinates and use Kabsch
            Xi = np.matmul(Ri, xworld) + ti
            Xj = np.matmul(Rj, xworld) + tj
            Xi = Xi.reshape((3, -1))
            Xj = Xj.reshape((3, -1))
            R_pair_ij, t_pair_ij = kabsch(Xi, Xj)
            R_pair[:, :, i, j] = R_pair_ij
            t_pair[:, i, j] = t_pair_ij
            log.VLOG(3, 'Pairwise rotation matrix R(%d, %d) = \n %s' % (i, j, R_pair_ij))
            log.VLOG(3, 'Pairwise translation vector R(%d, %d) = %s' % (i, j, t_pair_ij))

    log.info("Kabsch complete. Now minimizing...")

    ##################### Solve minimization problems for Rotation and Translation Estimates ####################

    # Solve linear least square problem to minimize translation vectors of all cameras.
    # This array is contructed here per pair of cameras and later resrtcutured to set up the linear minimization problem
    # Ax - b.
    log.info("Minimizing for first estimates of translation vectors per camera.")
    A = np.zeros((3, 3 * ncams, ncams, ncams))
    b = np.zeros((3, ncams, ncams))

    # Construct expanded matrix expression for minimization.
    for i in range(0, ncams):
        for j in range(0, ncams):
            if i == j:
                continue
            Rij = R_pair[:, :, min(i, j), max(i, j)]
            tij = t_pair[:, min(i, j), max(i, j)]

            A[:, 3 * i: 3 * (i + 1), i, j] = -Rij
            A[:, 3 * j: 3 * (j + 1), i, j] = np.eye(3)

            b[:, i, j] = tij

    A = np.transpose(A.reshape((3 * ncams, -1), order='F'))
    b = b.reshape((-1, 1), order='F')

    log.VLOG(4, 'Minimization matrix A for translation vectors \n %s' % A)
    log.VLOG(4, 'Minimization vector b for translation vectors \n %s' % b)

    # We want to constrain only the translational vector for the first camera
    # Create a constraint array with a 3x3 identity matrix in the top left
    constraint_array = np.zeros([3 * ncams, 3 * ncams])
    constraint_array[0:3, 0:3] = np.eye(3)

    # Solve the minimization, requiring the first translational vector to be the zero vector

    # Initialize camera positions assuming the first camera is at the origin and the cameras
    # are uniformly spaced by horizontal a displacement vector and a vertical vector such as in
    # a rectangular grid of the dimensions to be specified.
    cam_hspacing = np.array([1, 0, 0])
    cam_vspacing = np.array([0, 1, 0])
    cam_num_row = 2
    cam_num_col = 2

    x0 = np.zeros((3 * ncams))
    for i in range(cam_num_row):
        for j in range(cam_num_col):
            cam_index = i * cam_num_col + j
            x0[3 * cam_index: 3 * (cam_index + 1)] = i * cam_vspacing + j * cam_hspacing

    def trans_cost(x):
        return np.linalg.norm(np.matmul(A, np.array(x)) - b)
    # trans_cost_deriv = autograd.grad(lambda *args: trans_cost(np.transpose(np.array(args))))

    res = scipy.optimize.minimize(lambda x: trans_cost(x), x0,
                                  constraints=(scipy.optimize.LinearConstraint(constraint_array, np.zeros(3 * ncams),
                                                                               np.zeros(3 * ncams))),
                                                                               method = 'trust-constr')
    if res.success:
        log.info("Minimization for Translation Vectors Succeeded!")
        t_vals = res.x
    else:
        log.error('Minimization Failed for Translation Vectors!')
    # Translation vectors stored as columns.
    t_vals = t_vals.reshape((3, -1), order='F')

    for i in range(t_vals.shape[1]):
        log.VLOG(3, 'R(%d) = \n %s' % (i, t_vals[:, i]))

    log.info('Minimizing for tranlation vectors of cameras: %s', res.message)

    # Solve linear least square problem to minimize rotation matrices.
    log.info("Minimizing for first estimates of rotation matrices per camera.")
    A = np.zeros((9, 9 * ncams, ncams, ncams))

    # Construct expanded matrix expression for minimization.
    for i in range(0, ncams):
        for j in range(0, ncams):
            if i == j:
                continue
            Rij = R_pair[:, :, min(i, j), max(i, j)]

            A[:, 9 * i: 9 * (i + 1), i, j] = np.eye(9)

            A[:, 9 * j: 9 * (j + 1), i, j] = -np.kron(np.eye(3), Rij)

    A = np.transpose(A.reshape((9 * ncams, -1), order='F'))
    b = np.zeros(A.shape[0])

    log.VLOG(4, 'Minimization matrix A for rotation matrices \n %s' % A)
    log.VLOG(4, 'Minimization vector b for rotation matrices \n %s' % b)

    # We want to constrain only the rotation matrix for the first camera
    # Create a constraint array with a 9x9 identity matrix in the top left
    constraint_array = np.zeros([9 * ncams, 9 * ncams])
    constraint_array[0:9, 0:9] = np.eye(9)
    bound = np.zeros(9 * ncams)
    bound[0] = 1
    bound[4] = 1
    bound[8] = 1

    # Initialize all rotation matrices to identities assuming no camera is rotated with respect to another.
    x0 = np.zeros(9 * ncams)
    for i in range(ncams):
        x0[9 * i] = 1
        x0[9 * i + 4] = 1
        x0[9 * i + 8] = 1

    # Solve the minimization, requiring the first rotation matrix to be the identity matrix
    # TODO: figure out (if possible) a better initial guess

    res = scipy.optimize.minimize(lambda x: np.linalg.norm(np.matmul(A, x) - b), x0,
                                  constraints=(scipy.optimize.LinearConstraint(constraint_array, bound, bound)),
                                  method = 'trust-constr')
    if res.success:
        log.info("Minimization for Rotational Matrices Succeeded!")
        R_vals = res.x
    else:
        log.error('Minimization Failed for Rotational Matrices!')

    log.info('Minimizing for rotational matrices of cameras: %s', res.message)
    # Rotation matrices stored rows first
    R_vals = R_vals.reshape((3, 3, -1))

    for i in range(R_vals.shape[2]):
        log.VLOG(3, 'R(%d) = \n %s' % (i, R_vals[:, :, i]))

    log.info("Finding average rotation matrices and translational vectors from single camera calibration.")

    # Obtain average Rs and ts per images over cameras.
    R_images = np.sum(boardRotMats, axis=3).transpose() / ncams
    t_images = np.sum(boardTransVecs, axis=2) / ncams

    for i in range(R_images.shape[2]):
        log.VLOG(3, 'Average rotation matrix for Image %d = \n %s' % (i, R_vals[:, :, i]))
    for i in range(t_images.shape[1]):
        log.VLOG(3, 'Average translation vector for Image %d = \n %s' % (i, t_images[:, i]))
    ####################### Final Minimization to Yield All Parameters######################

    # K_c, R_c, R_n, t_c, t_n.

    # Pack all matrices into a very tall column vector for minimization
    min_vec_ini = np.append(cameraMats.reshape((-1), order='F'),
                            np.append(R_vals.reshape((-1), order='F'),
                                      np.append(R_images.reshape((-1), order='F'),
                                                np.append(t_vals.reshape((-1), order='F'),
                                                          t_images.reshape((-1), order='F')))))

    # Minimize, no more additional constraints.
    reproj_res = scipy.optimize.minimize(lambda x: reproj_min_func(planeData, cameraData, umeas, xworld, x),
                                         min_vec_ini)

    if reproj_res.success:
        log.info("Reprojection Minimization Succeeded!")
        cameraMats, rotationMatsCam, rotationMatsBoard, transVecsCam, transVecsBoard = unpack_reproj_min_vector(
            cameraData, planeData, reproj_res.x)
    else:
        log.error('Reprojection Minimization Failed!')

    return cameraMats, rotationMatsCam, transVecsCam


def reproj_min_func(planeData, cameraData, umeas, xworld, min_vec):
    """Total error function to be minimized for the final estimates of camera, rotation, and translation matrices.
    Input
    ------------------------
    min_vec         The compact flattened vector storing K_c, R_c, R_n, t_c, t_n in sequence.

    Returns
    ------------------------
    sum_of_error    Total error under the current estimate.
    """
    nX = planeData.nX
    nY = planeData.nY
    ncams = cameraData.ncams
    nplanes = planeData.ncalplanes

    # Total error of reprojections.
    sum_of_err = 0
    cameraMatrices, rotationMatricesCam, rotationMatricesBoard, transVecsCam, transVecsBoard = unpack_reproj_min_vector(
        cameraData, planeData, min_vec)
    for c in range(ncams):
        camMat = cameraMatrices[:, :, c]
        rotMatCam = rotationMatricesCam[:, :, c]
        transVecCam = transVecsCam[:, c]
        for n in range(nplanes):
            rotMatBoard = rotationMatricesBoard[:, :, n]
            transVecBoard = transVecsBoard[:, n]
            for i in range(nX - 1):
                for j in range(nY - 1):
                    # isolate only points related to this board's placement
                    # corners = [umeas[:, n * nX * nY + (i + m) * nX + j + n, c] for m in range(2) for n in range(2)]
                    img_coor = umeas[:, n * nX * nY + i * nX + j, c]
                    world_coor = xworld[:, nX * nY * n + i * nX + j]
                    # normal_factor = computeNormalFactor(corners, camMat, rotMatCam, transVecCam)
                    sum_of_err = sum_of_err + (
                        reproj_error(1, img_coor, world_coor, camMat, rotMatCam, rotMatBoard,
                                     transVecCam, transVecBoard)) ** 2
    return sum_of_err


def unpack_reproj_min_vector(cameraData, planeData, min_vec):
    """ Unpacks minimized column vector into K_c, R_c, R_n, t_c, t_n.

    Input
    --------------------
    min_vec     9*ncams + 9*ncams + 9*nimgs + 3*ncams + 3*nimgs x 1

    Returns
    --------------------
    cameraMatrices              3 x 3 x ncams: camera matrices for individual cameras
    rotationMatricesCam         3 x 3 x ncams: rotation matrix per camera
    rotationMatricesBoard       3 x 3 x nplanes: rotation matrix per image
    transVecCam                 3 x ncams: translational vector per camera
    transVecBoard               3 x nplanes: translational vector per image
    """
    nplanes = planeData.ncalplanes
    ncams = cameraData.ncams

    # Set up storage variable for extraction
    cameraMatrices = np.zeros((3, 3, ncams))
    rotationMatricesCam = np.zeros((3, 3, ncams))
    rotationMatricesBoard = np.zeros((3, 3, nplanes))
    transVecsCam = np.zeros((3, ncams))
    transVecsBoard = np.zeros((3, nplanes))

    # Keep track of offset in indexing into min_vec
    min_vec_counter = 0

    # Extract camera matrices
    for i in range(ncams):
        cameraMatrices[:, :, i] = min_vec[9 * i: 9 * (i + 1)].reshape((3, 3))
    min_vec_counter = 9 * ncams

    # Extract rotation matrices for cameras
    for i in range(ncams):
        rotationMatricesCam[:, :, i] = min_vec[min_vec_counter + 9 * i: min_vec_counter + 9 * (i + 1)].reshape((3, 3))
    min_vec_counter = min_vec_counter + 9 * ncams

    # Extract rotation matrices for images
    for i in range(nplanes):
        rotationMatricesBoard[:, :, i] = min_vec[min_vec_counter + 9 * i: min_vec_counter + 9 * (i + 1)].reshape((3, 3))
    min_vec_counter = min_vec_counter + 9 * nplanes

    # Extract translation vectors for cameras
    for i in range(ncams):
        transVecsCam[:, i] = min_vec[min_vec_counter + 3 * i: min_vec_counter + 3 * (i + 1)]
    min_vec_counter = min_vec_counter + 3 * ncams

    # Extract rotation matrices for images
    for i in range(nplanes):
        transVecsBoard[:, i] = min_vec[min_vec_counter + 3 * i: min_vec_counter + 3 * (i + 1)]
    min_vec_counter = min_vec_counter + 3 * nplanes

    return cameraMatrices, rotationMatricesCam, rotationMatricesBoard, transVecsCam, transVecsBoard


def proj_red(ray_trace_vec, k=1):
    """Reduce an augmented 3D image vector, the ray tracing vector, to the image position.
    """
    return np.array([[ray_trace_vec[0], ray_trace_vec[1]]]).transpose() / k


def jacobian2(vec_fxn):
    """returns determinant of the jacobian of the 2-variable vec_fxn
    """
    dim = 2
    rows = np.array([lambda y, x: 1, lambda y, x: 1])
    for i in range(dim):
        rows[i] = autograd.jacobian(vec_fxn, i)
    return lambda x, y: rows[0](x, y)[0] * rows[1](x, y)[1] - rows[0](x, y)[1] * rows[1](x, y)[0]


def quadrilateralIntegration(corners, exp):
    """
    Integrates an expression over the quadrilateral defined by the corners
    :param corners: an array of four tuples of the form (x, y), defining the corners of the quadrilateral
    :param exp: the expression that gets integrated over the area of the quadrilateral
    :return:
    """
    corners = sorted(corners, key=lambda x: x[1])
    bottom = corners[:2]
    top = corners[2:]

    corners = sorted(corners, key=lambda x: x[0])
    left = corners[:2]
    right = corners[2:]

    def line(x, side):
        if side[1][0] == side[0][0]:
            return 0
        return (side[1][1] - side[0][1]) / (side[1][0] - side[0][0]) * (x - side[0][0]) + side[0][1]

    def top_line(x): return line(x, top)

    def bottom_line(x): return line(x, bottom)

    def left_line(x): return line(x, left)

    def right_line(x): return line(x, right)

    # note that corners are currently sorted from left to right and are of the form (x, y)
    left_lines = (bottom_line, left_line) if corners[0][1] < corners[1][1] else (left_line, top_line)
    left_triangle = (0, 0) if corners[0][0] == corners[1][0] else scipy.integrate.dblquad(exp, corners[0][0],
                                                                                          corners[1][0], *left_lines)

    quad = scipy.integrate.dblquad(exp, corners[1][0], corners[2][0], bottom_line, top_line)

    right_lines = (bottom_line, right_line) if corners[3][1] < corners[2][1] else (right_line, top_line)
    right_triangle = (0, 0) if corners[2][0] == corners[3][0] else scipy.integrate.dblquad(exp, corners[2][0],
                                                                                           corners[3][0], *right_lines)
    return left_triangle[0] + quad[0] + right_triangle[0]


def computeNormalFactor(corners, camMat, rotMatCam, transVecCam):
    """
    Computes normal factor from Muller eq 6 (Appendix C)
    :param corners: list of four tuples in order (x, y) representing four corners of a square on the checkerboard
    :param camMat: 3x3 Camera matrix
    :param rotMatCam: 3x3 rotation matrix
    :param transVecCam: 3x1 translation vector
    :return: normal factor
    """
    proj = lambda y, x: np.matmul(camMat, np.matmul(np.column_stack((rotMatCam, transVecCam)),
                                                    np.transpose(np.array([x, y, 0, 1]))))

    # Jacobian function.
    jac = jacobian2(proj)

    return quadrilateralIntegration(corners, jac)


def reproj_error(normal_factor, img_coor, world_coor, camMatrix, rotMatrixCam, rotMatrixBoard, transVecCam,
                 transVecBoard):
    """
    Compute the reprojection error term for a given pair of camera and image. See paper (Muller 2019) formula (6).
    :param normal_factor:   normalized area in pixels of a checkerboard tile
    :param img_coor:        image coordinate, from umeas
    :param world_coor:      world coordinate
    :param camMatrix:       3 x 3: camera matrix for this camera
    :param rotMatrixCam:    3 x 3: rotation matrix for this camera
    :param rotMatrixBoard:  3 x 3: rotation matrix for this image
    :param transVecCam:     3: translational vector for this camera
    :param transVecBoard:   3: translational vector for this image
    :return: the reprojection error
    """

    # TODO scaling parameter k for principal optical axis assumed here.

    # [R_c t_c].
    transMatCam = np.column_stack((rotMatrixCam, transVecCam))
    # Corresponds to eq(6) in Muller paper. 4x4 matrix with image rot matrices and trans vectors
    transMatImg = np.column_stack((rotMatrixBoard, transVecBoard))
    rowToAdd = np.zeros(4)
    rowToAdd[3] = 1
    transMatImg = np.row_stack((transMatImg, rowToAdd))
    aug_world_coor = np.transpose(np.append(world_coor, 1))

    # Compute matrix multiplication
    product = np.matmul(camMatrix, np.matmul(transMatCam, np.matmul(transMatImg, aug_world_coor)))

    # Compute reprojection error term.
    # TODO revisit normal factor from Muller paper. Setting to 1 because for our purposes, volume depth is shallow.
    normal_factor = 1
    return 1 / np.sqrt(normal_factor) * np.sqrt(np.sum((img_coor - proj_red(product)) ** 2))


def saveCalibData(camData, X):
    """
    Save and display calibration data
    :param cparams:     parameters from which the matrices were contructed
    :param X:           world points
    :return: f - file to which data was saved (closed)
    plots world points and camera locations, with a sample of the wall and saves data on the experiment path
    """

    # plot world points and camera locations
    fig = plt.figure('Camera and Grid Locations', figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    # include a sample of the tank wall for context
    grid, = ax.plot(X[0, :], X[1, :], X[2, :], 'b.', label='Grid Points')
    cams, = ax.plot(cparams[0, :], cparams[1, :], cparams[2, :], 'r+', label='Cameras')
    plt.legend(handles=[grid, cams, wall])
    plt.show()

    return plt


if __name__ == '__main__':
    # read config file flag passed from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='relative path to config file', type=str)
    parser.add_argument('-v', '--verbosity', help='verbosity level for file prints (1 through 4 or DEBUG, INFO, etc.)',
                        type=str, default="1")
    parser.add_argument('-l', '--logType', help='style of log print messages (cpp (default), pretty)', type=str,
                        default="cpp")
    args = parser.parse_args()
    configPath = args.config_file
    log = logger.getLogger(__file__, args.verbosity, args.logType)

    import refCalib as rC  # need to import here for logging purposes

    # read and parse config file
    planeData, cameraData, sceneData, tolerances, calImgs, exptPath, camIDs, _ = rC.parseConfigFile(configPath)

    # find corners and pix_phys
    umeas = rC.findCorners(planeData, cameraData.ncams, exptPath, imgs=calImgs)
    # find pixel scaling using passed in parameters
    pix_phys = rC.getScale(planeData, umeas)

    # perform single camera calibration to get initial calibration matrices
    # and board geometric changes
    nX = planeData.nX
    nY = planeData.nY
    nplanes = planeData.ncalplanes
    xworld = np.array([[i, j, 0] for i in range(nX) for j in range(nY) for _ in range(nplanes)]).astype('float32')
    camMatrix, boardRotMat, boardTransVec = singleCamCalib(umeas, xworld, planeData, cameraData)

    cameraMats, rotationMatsCam, transVecsCam = multiCamCalib(umeas, np.transpose(xworld), camMatrix, boardRotMat,
                                                              boardTransVec,
                                                              planeData, cameraData)

    # perform multi camera calibration
    # TODO: check inputs after multiCamCalib is fully written
    # TODO: will need more variable than just x
    # x = multiCamCalib(umeas, camMatrix, boardRotMat, boardTransVec, planeData, cameraData)

    # TODO: Change saved data according to what multiCamCalib returns (we should probably try to make it return these
    #  though)
    xworld = np.array([[i * 10, j * 10, k * 5] for i in range(nX) for j in range(nY) for k in range(nplanes)]).astype(
        'float32')
    f = saveCalibData(cameraMats, rotationMatsCam, transVecsCam, xworld)
    print('\nData saved in ' + str(f))
