import os
import sys
import glob
#import tiffcapture as tc #using jpg for now
import numpy as np
import cv2
import numpy.linalg as lin
import math
import itertools
import warnings
import scipy.optimize
import matplotlib.pyplot as plt
import time
import configparser
import argparse

# This code completes a "freeform" camera calibration that allows us to place the board without much constraint in the calibration images.

def singleCamCalib(umeas, xworld, planeData, cameraData):
    """
    Inputs
    -------------------------------------------
    umeas              2 x nX*nY*nplanes x ncams array of image points in each camera
    xworld             3 x nX*nY*nplanes matrix of all world points
    planeData          struct of image related parameters
    cameraData         struct of camera related parameters
    
    Returns
    -------------------------------------------
    cameraMats         3 x 3 x ncams: camera matrices for individual cameras
    boardRotMats       3 x 3*nplanes x ncams: board rotation matrix per image per camera 
    boardTransVecs     3 x nplanes x ncams: board translational vector per image per camera
    """

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
    for i in range(0, ncams):

        # calculate initial camera matrix for this camera 
        # use all world points but only image points in this camera
        camUmeas = umeas[:, :, i]
        camMatrix = cv2.initCameraMatrix2D(xworld, camUmeas, nX*nY, imageSize, aspectRatio)

        # iterate through each board placement to find rotation matrix and rotation vector
        for n in range(0, nplanes):

            # isolate only points related to this board's placement
            currWorld = xworld[:, nX*nY*n: nX*nY*(n+1)]
            currImage = umeas[:, nX*nY*n: nX*nY*(n+1), i]

            # find the board's rotation matrix and translation vector
            _, rotMatrix, transVector = cv2.solvePnP(currWorld, currImage, camMatrix, np.zeroes((8,1), dtype='float32')) 

            # add board values to storage variables
            boardRotMats[:, :, n, i] = rotMatrix
            boardTransVecs[:, n, i] = transVector 

        # add camera matrix to storage variable
        cameraMats[:, :, i] = camMatrix

    return [cameraMats, boardRotMats, boardTransVecs]


def kabsch(X1, X2):
    """Returns pairwise rotation matrix and translation vector by 
       computing the cross-covariance matrix given two set of world points.
       
       Inputs
       ------------------------
        # X1: The world coordinates of the first camera stacked as columns.
        # X2: The world coordinates of the second camera stacked as columns.
    """
    
    x1m = X1.mean(axis = 1)
    x2m = X2.mean(axis = 1)

    X1_centered = X1 - np.transpose(x1m)
    X2_centered = X2 - np.transpose(x2m)

    A = np.matmul(X2_centered, np.transpose(X1_centered))
    
    U, S, Vtrans = np.linalg.svd(A)

    # calculates variable to correct rotation matrix if necessary
    d = np.det(np.matmul(U, Vtrans))
    D = np.eye(3)
    D[2,2] = d

    R_12 = np.matmul(np.matmul(U,D), Vtrans)
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
    
    R = (q[0]**2 - np.dot(qc, qc))*np.eye(3) + 2*np.matmul(np.transpose(qc), qc) + 2*q[0]*Q;
    R = R/normalizationFactor
    return R                                


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

    # Extract dimensional data.
    nX = planeData.nX
    nY = planeData.nY
    nplanes = planeData.ncalplanes
    ncams = cameraData.ncams
    
    # Storage variables.
    R_pair = np.zeros([3, 3, ncams, ncams])
    t_pair = np.zeros([3, ncams, ncams])
    

    for i in range(0, ncams):
        for j in range(i+1, ncams):
            # TODO: qInputMatrix to be defined as some collection of quarternions.
            # TODO: ti, tj is to be determined from input data.
            
            # Compute world coordinates used by kabsch.
            Ri = np.array([quaternionToRotMat(row) for row in np.transpose(qiInputMatrix)])
            Ri = Ri.reshape((1, -1, 3))[0]
            ti  = ti.reshape((-1, 1))
            Rj = np.array([quaternionToRotMat(row) for row in np.transpose(qjInputMatrix)])
            Rj = Rj.reshape((1, -1, 3))[0]
            tj  = tj.reshape((-1, 1))
            
            # Compute world coordinates and use Kabsch
            Xi = np.matmul(Ri, xworld) + ti
            Xj = np.matmul(Rj, xworld) + tj
            Xi.reshape((3, -1))
            Xj.reshape((3, -1))
            R_pair[:, :, i, j], t_pair[:, i, j] = kabsch(Xi, Xj)
    
    ##################### Solve minimization problems for Rotation and Translation Estimates ####################
    
    # Solve linear least square problem to minimize translation vectors of all cameras.
    # This array is contructed here per pair of cameras and later resrtcutured to set up the linear minimization problem
    # Ax - b.
    A = np.array(3, 3*ncams, ncams, ncams)
    b = np.array(3, ncams, ncams)
    
    # Construct expanded matrix expression for minimization.
    for i in range(0, ncams):
        for j in range(0, ncams):
            if (i == j):
                continue
            Rij = R_pair[:, :, min(i,j), max(i,j)]
            tij = t_pair[:, min(i,j), max(i,j)]
            
            A[:, 3*i: 3*(i+1), i, j] = -Rij
            A[:, 3*j: 3*(j+1), i, j] = np.eye(3)
            
            b[:, i, j] = tij

    A = np.transpose(A.reshape((3, -1), order = 'F'))
    b = b.reshape((-1, 1), order = 'F')
    
    # We want to constrain only the translational vector for the first camera
    # Create a constraint array with a 3x3 identity matrix in the top left
    constraint_array = np.zeros([3*ncams, 3*ncams])
    constraint_array[0:3, 0:3] = np.eye(3)

    # Solve the minimization, requiring the first translational vecotr to be the zero vector
    
    # Initialize camera positions assuming the first camera is at the origin and the cameras
    # are uniformly spaced by horizontal a displacement vector and a vertical vector such as in
    # a rectangular grid of the dimensions to be specified.
    cam_hspacing = np.array([1, 0 ,0])
    cam_vspacing = np.array([0, 1 ,0])
    cam_num_row = 3
    cam_num_col = 3
    
    x0 = np.zeros(3*ncams)
    for i in range(cam_num_row):
        for j in range(cam_num_col):
            cam_index = i*cam_num_col + j
            x0[3*cam_index: 3*(cam_index + 1)] = i * cam_vspacing + j * cam_vspacing

    res = scipy.optimize.minimize(lambda x: np.matmult(A, x) - b, x0,
                            constraints=(scipy.minimize.LinearConstraint(constraint_array, np.zeros(3*ncams), np.zeros(3*ncams))))
    if res.success:
        t_vals = res.x
    else:
        print('Minimization Failed for Translation Vectors!')
    # Translation vectors stored as columns.
    t_vals = t_vals.reshape((3, -1), order = 'F')
    
    # Solve linear least square problem to minimize rotation matrices.
    A = np.array(9, 9*ncams, ncams, ncams)
    
    # Construct expanded matrix expression for minimization.
    for i in range(0, ncams):
        for j in range(0, ncams):
            if (i == j):
                continue
            Rij = R_pair[:, :, min(i,j), max(i,j)]
            
            A[:, 9*i: 9*(i+1), i, j] = np.eye(9)
            
            A[:, 9*j: 9*(j+1), i, j] = -np.kron(np.eye(3), Rij)
            
    A = np.transpose(A.reshape((9, -1), order = 'F'))
    b = np.zeros(A.shape[0])
    
    # We want to constrain only the rotation matrix for the first camera
    # Create a constraint array with a 9x9 identity matrix in the top left
    constraint_array = np.zeros([9*ncams, 9*ncams])
    constraint_array[0:9, 0:9] = np.eye(9)
    bound = np.zeros(9*ncams)
    bound[0] = 1
    bound[4] = 1
    bound[8] = 1
    
    # Initialize all rotation matrices to identities assuming no camera is rotated with respect to another.
    x0 = np.zeroes(9*ncams)
    for i in range(ncams):
        x0[9*i] = 1
        x0[9*i + 4] = 1
        x0[9*i + 8] = 1
    
    # Solve the minimization, requiring the first rotation matrix to be the identity matrix
    # TODO: figure out (if possible) a better initial guess
    
    res = scipy.optimize.minimize(lambda x: np.matmult(A, x) - b, x0,
                            constraints=(scipy.minimize.LinearConstraint(constraint_array, bound, bound)))
    if res.success:
        R_vals = res.x
    else:
        print('Minimization Failed for Rotational Matrices!')
    # Rotation matrices stored rows first
    R_vals = R_vals.reshape((3, 3, -1))
    
    # Obtain average Rs and ts per images over cameras.
    R_images = np.sum(boardRotMats, axis = 3).transpose() / ncams
    t_images = np.sum(boardTransVecs, axis = 2) / ncams
    
    ####################### Final Minimization to Yield All Parameters######################
    
    # K_c, R_c, R_n, t_c, t_n.
    
    # Pack all matrices into a very tall column vector for minimization
    min_vec_ini = np.append(cameraMats.reshape((-1), order = 'F'),
                            np.append(R_vals.reshape((-1), order = 'F')),
                            np.append(R_images.reshape((-1), order = 'F')),
                            np.append(t_vals.reshape((-1), order = 'F'), t_images.reshape((-1), order = 'F')))
    
    # Minimize, no more additional constraints.
    reproj_res = scipy.optimize.minimize(lambda x: reproj_min_func(planeData, cameraData, umeas, xworld, x), min_vec_ini)
    
    if reproj_res.success:
        cameraMats, rotationMatsCam, rotationMatsBoard, transVecsCam, transVecsBoard = unpack_reproj_min_vector(reproj_res.x)
    else:
        print('Reprojection Minimization Failed!')
    
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
    cameraMatrices, rotationMatricesCam, rotationMatricesBoard, transVecsCam, transVecsBoard = unpack_reproj_min_vector(min_vec)
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
                    img_coor = umeas[:, n*nX*nY + i + j, c]
                    world_coor = xworld[:, nX*nY*n + i + j]
                    sum_of_err = sum_of_err + (reproj_error(normal_factor, img_coor, world_coor, camMat, rotMatCam, rotMatBoard, transVecCam, transVecBoard))**2
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
        cameraMatrices[:, :, i] = min_vec[9*i: 9*(i+1)].reshape((3, 3))
    min_vec_counter = 9*ncams
    
    # Extract rotation matrices for cameras
    for i in range(ncams):
        rotationMatricesCam[:, :, i] = min_vec[min_vec_counter + 9*i: min_vec_counter + 9*(i+1)].reshape((3, 3))
    min_vec_counter = min_vec_counter + 9*ncams
    
    # Extract rotation matrices for images
    for i in range(nplanes):
        rotationMatricesBoard[:, :, i] = min_vec[min_vec_counter + 9*i: min_vec_counter + 9*(i+1)].reshape((3, 3))
    min_vec_counter = min_vec_counter + 9*nplanes
    
    # Extract translation vectors for cameras
    for i in range(ncams):
        transVecsCam[:, i] = min_vec[min_vec_counter + 3*i: min_vec_counter + 3*(i+1)]
    min_vec_counter = min_vec_counter + 3*ncams
    
    # Extract rotation matrices for images
    for i in range(nplanes):
        transVecsBoard[:, i] = min_vec[min_vec_counter + 3*i: min_vec_counter + 3*(i+1)]
    min_vec_counter = min_vec_counter + 3*nplanes
    
    return cameraMatrices, rotationMatricesCam, rotationMatricesBoard, transVecsCam, transVecsBoard
    
    
def proj_red(ray_trace_vec, k):
    """Reduce an augmented 3D image vector, the ray tracing vector, to the image position.
    """
    return np.array([[ray_trace_vec[0], ray_trace_vec[1]]]).transpose()/k


def reproj_error(normal_factor, img_coor, world_coor, camMatrix, rotMatrixCam, rotMatrixBoard, transVecCam, transVecBoard):
    """Compute the reprojection error term for a given pair of camera and image. See paper (Muller 2019) formula (6).
    
    Inputs
    -----------------------------------
    normal_factor               normalized area in pixels of a checkerboard tile
    img_coor                    image coordinate, from umeas
    world_coor                  world coordinate
    cameraMatrix                3 x 3: camera matrix for this camera
    rotMatricesCam              3 x 3: rotation matrix for this camera
    rotMatricesBoard            3 x 3: rotation matrix for this image
    transVecCam                 3: translational vector for this camera
    transVecBoard               3: translational vector for this image
    
    Returns
    ------------------------------------
    the reprojection error
    """
    
    # TODO scaling paramter k for principal optical axis assumed here.
    
    # [R_c t_c].
    transMatCam = np.column_stack((rotMatrixCam, transVecCam))
    # Corresponds to eq(6) in Muller paper. 4x4 matrix with image rot matrices and trans vectors
    transMatImg = np.column_stack((rotMatrixBoard, transVecBoard))
    rowToAdd = np.zeros(4)
    rowToAdd[3] = 1
    transMatImg = np.row_stack((transMatImg, rowToAdd))
    aug_world_coor = np.row_stack((world_coor, np.array([[1]])))
    
    # Compute matrix multiplication
    product = np.matmul(camMatrix, np.matmul(transMatCam, np.matmul(transMatImg, aug_world_coor)))

    # Compute reprojection error term.
    return 1/np.sqrt(normal_factor)*np.sqrt(np.sum((img_coor - proj_red(product))**2))
