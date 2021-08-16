import argparse
import os
import sys
from time import sleep
import autograd
# import tiffcapture as tc #using jpg for now
import autograd.numpy as np
import cv2
import scipy
import scipy.integrate
import scipy.optimize
from scipy.sparse import lil_matrix
import cvxpy as cp
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, report_fit
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'python/lib/'))
import logger

#from test import interpretResults

if not __name__ == "__main__":
    import traceback

    filename = traceback.format_stack()[0]
    log = logger.getLogger(filename.split('"')[1], False, False)


def fitRotMat(Q):
    """
    fit best rotation matrix to input matrix
    Parameters
    ----------
    Q   input matrix

    Returns
    -------
    rotation matrix
    """
    U, S, Vt = np.linalg.svd(Q)

    # constrain to be proper
    D = np.eye(S.shape[0])
    D[-1, -1] = np.linalg.det(np.matmul(U, Vt))

    return np.matmul(U, np.matmul(D, Vt)), S


def singleCamCalibMat(umeas, xworld, planeData, cameraData):
    # set up implementation variables
    nX = planeData.nX
    nY = planeData.nY
    nplanes = planeData.ncalplanes
    ncams = cameraData.ncams
    imageSize = (cameraData.sX, cameraData.sY)
    aspectRatio = cameraData.sX / cameraData.sY

    # set up storage variables
    cameraMats = np.zeros([3, 3, ncams])
    rcb = np.zeros((3, 3, nplanes, ncams))
    tcb = np.zeros((3, nplanes, ncams))
    repErr = np.zeros((nX*nY, nplanes, ncams))
    
    # try using initial guess for param (eventually get from config file)
    init_f_value = 9000;
    #init_f_value = 2500;
                        
    for c in range(ncams):
        
        #initialize comparison variables
        objpts = [] # world coordinates
        imgpts = [] # image coordinates
        
        camUmeas = np.transpose(umeas[:, :, c]).astype('float32')
        
        for n in range(nplanes):
            x = camUmeas[n * nX * nY:(n + 1) * nX * nY, :]
            X = xworld[n * nX * nY:(n + 1) * nX * nY, :]
            objpts.append(X)
            imgpts.append(x)

        #camMatrix = cv2.initCameraMatrix2D(objpts, imgpts, imageSize, 0)
        camMatrix = np.zeros((3,3))
        camMatrix[0,0] = init_f_value
        camMatrix[1,1] = init_f_value
        camMatrix[0,2] = imageSize[0]*0.5
        camMatrix[1,2] = imageSize[1]*0.5
        camMatrix[2,2] = 1
        ret, camMatrix, _, rVec, tVec = cv2.calibrateCamera(objpts, imgpts, imageSize,camMatrix,None,rvecs=None,
                      tvecs=None,flags=cv2.CALIB_FIX_ASPECT_RATIO+cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_FIX_PRINCIPAL_POINT)
        
        cameraMats[:, :, c] = camMatrix
        log.VLOG(4, "with Intrinsic Parameter Matrix %s", camMatrix)
        
        for n in range(nplanes):
            x = camUmeas[n * nX * nY:(n + 1) * nX * nY, :]
            X = xworld[n * nX * nY:(n + 1) * nX * nY, :]

            # rotation matrix from rotation vector
            R, _ = cv2.Rodrigues(rVec[n].ravel())
            rcb[:, :, n, c] = R

            # translation vector
            tcb[:, n, c] = tVec[n].ravel()
            
            # Reprojection error (currently for debugging only)
            testPts, _ = cv2.projectPoints(X,R,tVec[n],camMatrix,None)
            testPts = np.squeeze(testPts)
            tmpErr = np.linalg.norm (testPts - x, axis=1)
            repErr[:, n, c] = tmpErr
            
    return cameraMats, rcb, tcb


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
    boardRotMats       3 x 3 x nplanes x ncams: board rotation matrix per image per camera
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
            _, rotvec, transVector = cv2.solvePnP(currWorld, currImage, camMatrix,
                                                     np.zeros((8, 1), dtype='float32'))
            # Convert rotation vector to rotation matrix.
            rotMatrix, _ = cv2.Rodrigues(rotvec)

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
    nplanes = planeData.ncalplanes

    # Storage variables.
    R_pair = np.zeros([3, 3, ncams, ncams])
    t_pair = np.zeros([3, ncams, ncams])

    xworld = np.transpose(xworld)

    for i in range(0, ncams):
        for j in range(0, ncams):
            if i ==j:
                t_pair[:, i, j] = np.zeros((3))
                R_pair[:,:,i,j] = np.eye(3,3)
                continue
            log.VLOG(2, 'Correlating cameras (%d, %d)' % (i, j))

            # Compute world coordinates used by kabsch.
            Ri = np.concatenate(np.moveaxis(boardRotMats[:, :, :, i], 2, 0), axis=0)
            ti = boardTransVecs[:, :, i].reshape((-1, 1), order='F')
            Rj = np.concatenate(np.moveaxis(boardRotMats[:, :, :, j], 2, 0), axis=0)
            tj = boardTransVecs[:, :, j].reshape((-1, 1), order='F')

            # Compute world coordinates and use Kabsch
            Xi = np.matmul(Ri, xworld) + ti
            Xj = np.matmul(Rj, xworld) + tj
            Xi = Xi.reshape((3, -1),order='F')
            Xj = Xj.reshape((3, -1),order='F')
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
            #Rij = R_pair[:, :, min(i, j), max(i, j)]
            #tij = t_pair[:, min(i, j), max(i, j)]
            Rij = R_pair[:, :, i, j]
            tij = t_pair[:, i, j]

            A[:, 3 * i: 3 * (i + 1), i, j] = -Rij
            A[:, 3 * j: 3 * (j + 1), i, j] = np.eye(3)

            b[:, i, j] = tij

    A = np.concatenate(np.concatenate(np.moveaxis(A, (2, 3), (0, 1)), axis=0), axis=0)
    b = b.reshape((-1, 1), order='F')

    log.VLOG(4, 'Minimization matrix A for translation vectors \n %s' % A)
    log.VLOG(4, 'Minimization vector b for translation vectors \n %s' % b)

    # We want to constrain only the translational vector for the first camera
    # Create a constraint array with a 3x3 identity matrix in the top left
    constraint_array = np.zeros([3 * ncams, 3 * ncams])
    constraint_array[0:3, 0:3] = np.eye(3)

    # Solve the minimization, requiring the first translational vector to be the zero vector

    # Initialize camera positions assuming the first camera is at the origin
    x0 = t_pair[:,:,0]
    x0 = np.reshape(x0,(-1,1), order='F')

    #x0 = np.zeros((3 * ncams))
    #for i in range(cam_num_row):
    #    for j in range(cam_num_col):
    #        cam_index = i * cam_num_col + j
    #        x0[3 * cam_index: 3 * (cam_index + 1)] = i * cam_vspacing + j * cam_hspacing

    def trans_cost(x):
        return np.linalg.norm(np.matmul(A, np.array(x)) - b)
    # trans_cost_deriv = autograd.grad(lambda *args: trans_cost(np.transpose(np.array(args))))

    # Construct the problem.
    x = cp.Variable((3 * ncams, 1))
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [constraint_array @ x == np.zeros((3 * ncams, 1))]
    prob = cp.Problem(objective, constraints)

    print("Optimal value", prob.solve())

    if prob.status == cp.OPTIMAL:
        log.info("Minimization for Translation Vectors Succeeded!")
        print('Optimized translation error: ', trans_cost(x.value))
        t_vals = x.value
    else:
        log.error('Minimization Failed for Translation Vectors!')
        return
    # Translation vectors stored as columns.
    t_vals = np.transpose(t_vals.reshape((-1, 3)))

    for i in range(t_vals.shape[1]):
        log.VLOG(3, 't(%d) = \n %s' % (i, t_vals[:, i]))

    # log.info('Minimizing for translation vectors of cameras: %s', res.message)

    # Solve linear least square problem to minimize rotation matrices.
    log.info("Minimizing for first estimates of rotation matrices per camera.")
    A = np.zeros((9, 9 * ncams, ncams, ncams))

    # Construct expanded matrix expression for minimization.
    for i in range(0, ncams):
        for j in range(0, ncams):
            if i == j:
                continue
            Rij = R_pair[:, :, i, j]

            A[:, 9 * i: 9 * (i + 1), i, j] = np.eye(9)

            A[:, 9 * j: 9 * (j + 1), i, j] = -np.kron(np.eye(3), Rij)

    A = np.concatenate(np.concatenate(np.moveaxis(A, (2, 3), (0, 1)), axis=0), axis=0)
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

    # Construct the problem.
    x = cp.Variable(9 * ncams)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [constraint_array @ x == bound]
    prob = cp.Problem(objective, constraints)

    print("Optimal value", prob.solve())

    print('Minimization status: ', prob.status)
    if prob.status == cp.OPTIMAL:
        log.info("Minimization for Rotational Matrices Succeeded!")
        #print('Optimized rotation error: ', rotCost(x.value))
        R_vals = x.value
    else:
        log.error('Minimization Failed for Rotational Matrices!')
        return

    # Rotation matrices stored rows first
    R_vals = np.moveaxis(R_vals.reshape(-1, 3, 3), 0, 2)

    # Fit rotation matrices from optimized result.
    for i in range(ncams):
        R_vals[:, :, i], _ = fitRotMat(R_vals[:, :, i])
        log.VLOG(3, 'R(%d) = \n %s' % (i, R_vals[:, :, i]))

    log.info("Finding average rotation matrices and translational vectors from single camera calibration.")

    # Obtain average Rs and ts per images over cameras.
    R_images = np.sum(boardRotMats, axis=3) / ncams
    t_images = np.sum(boardTransVecs, axis=2) / ncams

    for i in range(R_images.shape[2]):
        log.VLOG(3, 'Average rotation matrix for Image %d = \n %s' % (i, R_images[:, :, i]))
    for i in range(t_images.shape[1]):
        log.VLOG(3, 'Average translation vector for Image %d = \n %s' % (i, t_images[:, i]))

    ####################### Final Minimization to Yield All Parameters######################

    # K_c, R_c, R_n, t_c, t_n.
    # cameraMats, R_vals, t_vals = interpretResults('../../../FILMopenfv-samples/sample-data/pinhole_calibration_data/calibration_results/results_000001438992543.txt')

    # Pack all matrices into a very tall column vector for minimization
    min_vec_ini = np.append(cameraMats.reshape((-1)),
                            np.append(R_vals.reshape((-1)),
                                      np.append(R_images.reshape((-1)),
                                                np.append(t_vals.reshape((-1)),
                                                          t_images.reshape((-1))))))

    planeVecs = {}

    # Set up objective function and Jacobian sparsity structure for minimization.
    def reproj_obj_lsq(min_vec):
        # Compute the reprojection error at each intersection per each camera per each image.

        cameraMats, rotMatsCam, rotMatsBoard, transVecsCam, transVecsBoard = unpack_reproj_min_vector(
            cameraData, planeData, min_vec)

        err = np.zeros(ncams * nplanes * nX * nY)

        # Set pixel normalization factor to 1 here.
        normal_factor = 1

        for c in range(ncams):
            for n in range(nplanes):
                for i in range(nX):
                    for j in range(nY):
                        err[c * nplanes * nX * nY + n * nX * nY + i * nY + j] = \
                            reproj_error(normal_factor, umeas[:, n * nX * nY + i * nY + j, c],
                                         xworld[:, nX * nY * n + i * nY + j], cameraMats[:, :, c],
                                         rotMatsCam[:, :, c], rotMatsBoard[:, :, n], transVecsCam[:, c],
                                         transVecsBoard[:, n])
        # print('Mean Reprojection Error: ', np.linalg.norm(err))
        return err

    def bundle_adjustment_sparsity():
        m = ncams * nplanes * nX * nY
        n = min_vec_ini.shape[0]
        A = lil_matrix((m, n), dtype=int)

        num_nonzeros = np.sum([np.prod(x.shape[:-1]) for x in [cameraMats, R_vals, t_vals, R_images, t_images]])
        for c in range(ncams):
            # make boolean arrays for camera-indexed arrays
            cams = np.zeros(cameraMats.shape)
            cams[:, :, c] = np.ones(cameraMats.shape[:-1])
            rots = np.zeros(R_vals.shape)
            rots[:, :, c] = np.ones(R_vals.shape[:-1])
            ts = np.zeros(t_vals.shape)
            ts[:, c] = np.ones(t_vals.shape[:-1])

            cams = cams.reshape(-1)
            rots = rots.reshape(-1)
            ts = ts.reshape(-1)

            for n in range(nplanes):
                # make boolean arrays for plane-indexed arrays
                if n not in planeVecs:
                    rimgs = np.zeros(R_images.shape)
                    rimgs[:, :, n] = np.ones(R_images.shape[:-1])
                    timgs = np.zeros(t_images.shape)
                    timgs[:, n] = np.ones(t_images.shape[:-1])
                    planeVecs[n] = rimgs, timgs
                else:
                    rimgs, timgs = planeVecs[n]

                rimgs = rimgs.reshape(-1)
                timgs = timgs.reshape(-1)

                # create boolean array with changed values for jacobian
                x0 = np.append(cams.reshape((-1)),
                               np.append(rots.reshape((-1)),
                                         np.append(rimgs.reshape((-1)),
                                                   np.append(ts.reshape((-1)),
                                                             timgs.reshape((-1))))))

                # set changing values in jacobian
                A[nY * nX * (n + nplanes * c): nY * nX * (n + 1 + nplanes * c), x0.nonzero()] = np.ones(
                    (nY * nX, num_nonzeros))

        return A

    A = bundle_adjustment_sparsity()
    reproj_res = scipy.optimize.least_squares(reproj_obj_lsq, min_vec_ini, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-2)

    if reproj_res.success:
        finError = reproj_min_func(planeData, cameraData, umeas, xworld, reproj_res.x)
        print("Final error: {}".format(reproj_min_func(planeData, cameraData, umeas, xworld, reproj_res.x)))
        log.info("Reprojection Minimization Succeeded!")
        cameraMats, rotationMatsCam, rotationMatsBoard, transVecsCam, transVecsBoard = unpack_reproj_min_vector(
            cameraData, planeData, reproj_res.x)
    else:
        log.error('Reprojection Minimization Failed!')
        return

    return cameraMats, rotationMatsCam, rotationMatsBoard, transVecsCam, transVecsBoard, finError


error_dict = {}
areas = {}
error_list = []


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

    if str(min_vec) in error_dict:
        return error_dict[str(min_vec)]

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
            for i in range(nX):
                for j in range(nY):
                    # isolate only points related to this board's placement
                    try:
                        normal_factor = areas[(c, n, i, j)]
                    except KeyError:
                        if i != nX - 1 and j != nY - 1:
                            corners = [umeas[:, n * nX * nY + (i + m) * nY + j + n, c] for m in range(2) for n in
                                       range(2)]
                            fake_x = np.mean([corners[0][0], corners[1][0]]) - np.mean([corners[2][0], corners[3][0]])
                            fake_y = np.mean([corners[0][1], corners[2][1]]) - np.mean([corners[1][1], corners[3][1]])
                            normal_factor = np.abs(fake_x * fake_y)
                            areas[(c, n, i, j)] = normal_factor
                        else:
                            normal_factor = areas[(c, n, min(i, nX - 2), min(j, nY - 2))]

                    img_coor = umeas[:, n * nX * nY + i * nY + j, c]
                    world_coor = xworld[:, nX * nY * n + i * nY + j]
                    # normal_factor = computeNormalFactor(corners, camMat, rotMatCam, transVecCam)
                    sum_of_err = sum_of_err + (
                        reproj_error(normal_factor, img_coor, world_coor, camMat, rotMatCam, rotMatBoard,
                                     transVecCam, transVecBoard)) ** 2

    error_dict[str(min_vec)] = sum_of_err
    global error_list
    error_list += [(min_vec, sum_of_err)]

    def deriv(entry1, entry2):
        x1, err1 = entry1
        x2, err2 = entry2
        x1 = np.array(x1)
        x2 = np.array(x2)
        return (err1 - err2) / np.linalg.norm(x1 - x2)

    print('Error: ', sum_of_err)
    if len(error_list) > 1:
        print('derivative: ', deriv(error_list[-1], error_list[-2]))

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

    cameraMatrices = min_vec[0: 9 * ncams].reshape((3, 3, ncams))
    min_vec_counter = 9 * ncams

    rotationMatricesCam = min_vec[min_vec_counter: min_vec_counter + 9 * ncams].reshape((3, 3, ncams))
    min_vec_counter += 9 * ncams

    # Extract rotation matrices for images
    rotationMatricesBoard = min_vec[min_vec_counter: min_vec_counter + 9 * nplanes].reshape((3, 3, nplanes))
    min_vec_counter += 9 * nplanes

    # Extract translation vectors for cameras
    transVecsCam = min_vec[min_vec_counter: min_vec_counter + 3 * ncams].reshape(3, ncams)
    min_vec_counter += 3 * ncams

    # Extract rotation matrices for images
    transVecsBoard = min_vec[min_vec_counter: min_vec_counter + 3 * nplanes].reshape(3, nplanes)

    return cameraMatrices, rotationMatricesCam, rotationMatricesBoard, transVecsCam, transVecsBoard


def proj_red(ray_trace_vec):
    """Reduce an augmented 3D image vector, the ray tracing vector, to the image position.
    """
    return np.array([ray_trace_vec[0]/ray_trace_vec[2], ray_trace_vec[1]/ray_trace_vec[2]]).transpose()


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
    Parameters
    ----------
    normal_factor   normalized area in pixels of a checkerboard tile
    img_coor        image coordinate, from umeas
    world_coor      world coordinate
    camMatrix       3 x 3: camera matrix for this camera
    rotMatrixCam    3 x 3: rotation matrix for this camera
    rotMatrixBoard  3 x 3: rotation matrix for this image
    transVecCam     3: translational vector for this camera
    transVecBoard   3: translational vector for this image

    Returns
    -------
    the reprojection error
    """

    # TODO scaling parameter k for principal optical axis assumed here.

    # [R_c t_c].
    transMatCam = np.column_stack((np.transpose(rotMatrixCam), transVecCam))
    #transMatCam = np.column_stack((rotMatrixCam, -transVecCam))
    # Corresponds to eq(6) in Muller paper. 4x4 matrix with image rot matrices and trans vectors
    transMatImg = np.column_stack((rotMatrixBoard, transVecBoard))
    rowToAdd = np.zeros(4)
    rowToAdd[3] = 1
    transMatImg = np.row_stack((transMatImg, rowToAdd))
    aug_world_coor = np.append(world_coor, 1)

    # Compute matrix multiplication
    product = np.matmul(camMatrix, np.matmul(transMatCam, np.matmul(transMatImg, aug_world_coor)))

    # Compute reprojection error term.
    # TODO revisit normal factor from Muller paper. Setting to 1 because for our purposes, volume depth is shallow.
    return 1 / np.sqrt(normal_factor) * np.linalg.norm(img_coor - proj_red(product))


def saveCalibData(exptPath, camnames, p, cparams, tplanes, sceneData, camData, finalError,
                      pix_phys, name):
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
    board, = ax.plot(tplanes[0, :], tplanes[1, :], tplanes[2, :], 'b.', label='Board Positions')
    cams, = ax.plot(cparams[0, :], cparams[1, :], cparams[2, :], 'r+', label='Cameras')
    plt.legend(handles=[board, cams])
    plt.show()
    
    ts = time.localtime()
    time_stamp = time.strftime('%H%M%m%d%y', ts)
    file = os.path.join(exptPath, name + time_stamp + '.dat')
    log.info(file)

    with open(file, 'w') as f:
        f.write('Pinhole Calibration performed: ' + time.strftime('%c', ts) + '\n')
        f.write(str(finalError) + '\n')
        f.write(str(camData.sX) + ' ' + str(camData.sY) + ' ' + str(pix_phys) + '\n')
        f.write(str(camData.ncams) + '\n')

        for c in range(len(camnames)):

            f.write(str(camnames[c]) + '\n')
            np.savetxt(f, p[:, :, c], delimiter=' ', fmt='%f')
            camParam = cparams[:, c]
            worldPoints = camParam[0:3]
            f.write(str(worldPoints[0]) + ' ' + str(worldPoints[1]) + ' ' +
                    str(worldPoints[2]) + '\n')
        f.write('0\n')  # calibration not refractive
        #f.write(str(scdata.zW) + ' ' + str(scdata.n[0]) + ' ' + str(scdata.n[1]) + ' ' +
        #        str(scdata.n[2]) + ' ' + str(scdata.tW) + '\n')
        #f.write('\n' + str(planeData.dX) + ' ' + str(planeData.dY) + ' ' +
        #        str(planeData.nX) + ' ' + str(planeData.nY) + '\n')
        #f.write(str(camData.so) + ' ' + str(camData.f) + ' ' + str(planeData.z0[-1]) + '\n')

        #f.write('Initial error: \n')
        #np.savetxt(f, errorLog[0], delimiter=', ', fmt='%f')
        #f.write('Final error: \n')
        #np.savetxt(f, errorLog[1], delimiter=', ', fmt='%f')

        return f  # file will close when with statement terminates

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
    dX = planeData.dX
    dY = planeData.dY
    nplanes = planeData.ncalplanes
    xworld = np.array(
        [[i * dX + 1, j * dY + 1, 0] for _ in range(nplanes) for i in range(nX) for j in range(nY)]).astype(
            'float32')
    camMatrix, boardRotMat, boardTransVec = singleCamCalibMat(umeas, xworld, planeData, cameraData)

    cameraMats, rotationMatsCam, rotationMatsBoard, transVecsCam, transVecBoard, finalError = multiCamCalib(umeas, 
                                                            xworld, camMatrix, boardRotMat, boardTransVec, planeData, cameraData)

    #print(rotationMatsCam)
    #print(transVecsCam)
    
    #construct P matrix
    P = np.zeros((3,4,cameraData.ncams))
    for c in range(cameraData.ncams):
        Rt = np.column_stack((np.transpose(rotationMatsCam[:,:,c]), transVecsCam[:,c]))
        P[:,:,c]=np.matmul(cameraMats[:,:,c],Rt)
    
    f = saveCalibData(exptPath, camIDs, P, transVecsCam, transVecBoard, sceneData, cameraData, finalError,
                      pix_phys, 'results_')
    log.info('\nData saved in ' + str(f))

    # TODO: Change saved data according to what multiCamCalib returns (we should probably try to make it return these
    #  though)
# xworld = np.array([[i * 10, j * 10, k * 5] for i in range(nX) for j in range(nY) for k in range(nplanes)]).astype(
#     'float32')
# f = saveCalibData(cameraMats, rotationMatsCam, transVecsCam, xworld)
# print('\nData saved in ' + str(f))
