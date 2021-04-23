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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'python/lib/'))
import logger

#Pseudocode:
# parse config file, get calibration images, get camera IDs, perform variable setups
# 
# find chessboard corners in all images and store them in variable called cor_all
#
# create "real world" coordinates using grid spacing and number of corners
#
# line up real world coordinates with corner of checkerboard image?
#       use planar_grid to world?
#       now have object points and image points
#
# perform single camera calibration:
#   cycle through all images in the camera
#   create an initial camera matrix using object and image points
#       use initCameraMatrix2D in opencv
#   find rotation and translation vectors for each checkerboard
#       store in a matrix in order
#
# perform multi camera calibration:
#   get first estimate of R and t for each camera
#       find rotation and translation vector for all checkerboard placements in two cameras, repeat for all camera pairings
#           put all points into [ x1 y1; x2 y2] order and store into matrices P and Q (corresponding to each camera)
#           calculate matrix H s.t H = P(^T)Q using summing notation
#           calculate SVD of matrix H
#           get R back by plugging V and U into a formula, giving us R^(P->Q)
#           get t by solving rigid body motion equation, giving us t^(P->Q)
#       for each camera, take a pairing(?) and solve a minimazation for t^P and R^P ??? appendix B doesn't make sense
#   estimate all checkerboard positions by averaging all camera's rot and trans vector for a checkerboard position
#   compute final K, R, and t for all cameras by minimizing reprojection error using our estimates
#  


def singleCamCalib(umeas, xworld, planeData, cameraData):
    #Inputs:
    # umeas is 2 x nX*nY*nplanes x ncams array of image points in each camera
    # xworld is 3 x nX*nY*nplanes of all world points
    #
    #Outputs:
    # cameraMats     - 3 x 3 x ncams that holds all camera calibration matrices
    # boardRotMats   - 3 x 3 x nplanes x ncams for each boar d in each camera
    # boardTransVecs - 3 x nplanes x ncams for each board in each camera

    log.info("Single camera calibrations")

    # set up implementation variables
    nX = planeData.nX
    nY = planeData.nY
    nplanes = planeData.ncalplanes
    ncams = cameraData.ncams
    imageSize = size(cameraData.sX, cameraData.sY)
    aspectRatio = cameraData.sX / cameraData.sY

    # set up storage variables 
    cameraMats = np.zeros([3, 3, ncams])
    boardRotMats = np.zeros([3, 3, nplanes, ncams])
    boardTransVecs = np.zeros([3, nplanes, ncams])

    # cycle through each camera 
    log.info("Cycle through each camera")
    for i in range(0, ncams):
        log.VLOG(2, "Calibrating camera %d", i)
        
        # calculate initial camera matrix for this camera 
        # use all world points but only image points in this camera
        camUmeas = umeas[:, :, i]
        camMatrix = initCameraMatrix2D(xworld, camUmeas, nX*nY, imageSize, aspectRatio)
        log.VLOG(4, "with Intrinsic Parameter Matrix %s", camMatrix)

        # iterate through each board placement to find rotation matrix and rotation vector
        for n in range(0, nplanes):
            log.VLOG(2, "on image %d", n)
            # isolate only points related to this board's placement
            currWorld = xworld[:, nX*nY*n: nX*nY*(n+1)]
            currImage = umeas[:, nX*nY*n: nX*nY*(n+1), i]

            # find the board's rotation and translation vector
            _, rotMatrix, transVector = solvePnP(currWorld, currImage, camMatrix, np.zeroes((8,1), dtype='float32')) 
            log.VLOG(4, "Rotation matrix for camera %d image %d: %s", i, n, rotMatrix)
            log.VLOG(4, "Translational vector for camera %d image %d: %s", i, n, transVector)
            
            # add board values to storage variables
            boardRotMats[:, :, n, i] = rotMatrix
            boardTransVecs[:, n, i] = transVector 

        # add camera matrix to storage variable
        cameraMats[:, :, i] = camMatrix

    return [cameraMats, boardRotMats, boardTransVecs]


def kabsch(X1, X2):
    # X1: The world coordinates of the first camera stacked as columns.
    # X2: The world coordinates of the second camera stacked as columns.
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
    t_12 = mean(x2 - np.matmul(R_12, x1), 1)

    return R_12, t_12

def quaternionToRotMat(q):
    # q is a 1 x 4 array.
    
    normalizationFactor = np.dot(q, q)
    
    # qc is a row vector
    qc = np.array([[q[1], q[2], q[3]]])
    # Cross operator matrix.
    Q = np.array([[0 -q[3] q[2]], [q[3] 0 -q[1]], [-q[2] q[1] 0]])
    
    R = (q[0]**2 - np.dot(qc, qc))*np.eye(3) + 2*np.matmul(np.transpose(qc), qc) + 2*q[0]*Q;
    R = R/normalizationFactor
    return R                                


def multiCamCalib(boardRotMats, boardTransVecs, planeData, cameraData):
    # loop over all camera pairs
    # call kabsch on each pair
    # store rotation matrices and translational vectors
    # Calculate terms in sum, add to running sum

    log.info("Multiple Camera Calibrations")

    # Extract dimensional data.
    nX = planeData.nX
    nY = planeData.nY
    nplanes = planeData.ncalplanes
    ncams = camreaData.ncams
    
    # Storage variables.
    R_pair = np.zeros([3, 3, ncams, ncams])
    t_pair = np.zeros([3, ncams, ncams])
    

    for i in range(0, ncams):
        for j in range(i+1, ncams):
            # TODO: qInputMatrix to be defined as some collection of quarternions.
            # TODO: ti, tj is to be determined from input data.
            
            log.VLOG(2, 'Correlating cameras (%d, %d)' % (i, j))
            
            # Compute world coordinates used by kabsch.
            Ri = np.array([quaternionToRotMat(row) for row in np.transpose(qiInputMatrix)])
            Ri = Ri.reshape((1, -1, 3))[0]
            ti  = ti.reshape((-1, 1))
            Rj = np.array([quaternionToRotMat(row) for row in np.transpose(qjInputMatrix)])
            Rj = Rj.reshape((1, -1, 3))[0]
            tj  = tj.reshape((-1, 1))
            
            # Compute world coordinates and use Kabsch
            Xi = np.matmul(Ri, xworld) + ti
            Xj = np.matmult(Rj, xworld) + tj
            Xi.reshape((3, -1))
            Xj.reshape((3, -1))
            R_pair_ij, t_pair_ij = kabsch(Xi, Xj)
            R_pair[:, :, i, j] = R_pair_ij
            t_pair[:, i, j] = t_pair_ij
            log.VLOG(3, 'Pairwise rotation matrix R(%d, %d) = \n %s' % (i, j, R_pair_ij))
            log.VLOG(3, 'Pairwise translation vector R(%d, %d) = %s' % (i, j, t_pair_ij))

    log.info("Kabsch complete. Now minimizing...")
    
    ##################### Solve minimization problems for Rotation and Translation Estimates ####################
    
    # Solve linear least square problem to minimize translation vectors of all cameras.
    # This array is contructed here per pair of cameras and later resrtcutured to set up the linear minimization problem
    # Ax - b = 0.
    log.info("Minimizing for first estimates of translation vectors per camera.")
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
            A[:, 3*j: 3*(j+1), i, j] = eye(3)
            
            b[:, i, j] = tij

    A = np.transpose(A.reshape((3, -1), order = 'F'))
    b = b.reshape((-1, 1), order = 'F')
    
    log.VLOG(4, 'Minimization matrix A for translation vectors \n %s' % (A))
    log.VLOG(4, 'Minimization vector b for translation vectors \n %s' % (b))
    
    # We want to constrain only the translational vector for the first camera
    # Create a constraint array with a 3x3 identity matrix in the top left
    constraint_array = np.zeros([3*ncams, 3*ncams])
    constraint_array[0:3, 0:3] = np.eye(3)

    # Solve the minimization, requiring the first translational vector to be the zero vector
    # TODO: figure out what x0 should be
    
    x0 = np.ones(3*ncams)
    x0[0:3] = np.zeros(3)
    res = scipy.optimize.minimize(lambda (x) np.matmult(A, x) - b, x0,
                            constraints=(scipy.minimize.LinearConstraint(constraint_array, np.zeros(3*ncams), np.zeros(3*ncams))))
    if res.success:
        log.info("Minimization for Translation Vectors Succeeded!")
        t_vals = res.x
    else:
        log.error('Minimization Failed for Translation Vectors!')
    # Translation vectors stored as columns.
    t_vals = t_vals.reshape((3, -1), order = 'F')
    
    for i in range(t_vals.shape[1]):
        log.VLOG(3, 'R(%d) = \n %s' % (i, t_vals[:, i]))
    
    log.info('Minimizing for tranlation vectors of cameras: %s', res.message)
    
    # Solve linear least square problem to minimize rotation matrices.
    log.info("Minimizing for first estimates of rotation matrices per camera.")
    A = np.array(9, 9*ncams, ncams, ncams)
    
    # Construct expanded matrix expression for minimization.
    for i in range(0, ncams):
        for j in range(0, ncams):
            if (i == j):
                continue
            Rij = R_pair[:, :, min(i,j), max(i,j)]
            
            A[:, 9*i: 9*(i+1), i, j] = np.eye(9)
            
            A[:, 9*j: 9*(j+1), i, j] = -np.kron(np.eye(3),Rij)
            
    A = np.transpose(A.reshape((9, -1), order = 'F'))
    b = np.zeros(A.shape[0])
    
    log.VLOG(4, 'Minimization matrix A for rotation matrices \n %s' % (A))
    log.VLOG(4, 'Minimization vector b for rotation matrices \n %s' % (b))
    
    # We want to constrain only the rotation matrix for the first camera
    # Create a constraint array with a 9x9 identity matrix in the top left
    constraint_array = np.zeros([9*ncams, 9*ncams])
    constraint_array[0:9, 0:9] = np.eye(9)
    bound = np.zeros(9*ncams)
    bound[0] = 1
    bound[4] = 1
    bound[8] = 1
    
    x0 = np.ones(9*ncams)
    x0[0:9] = np.zeros(9)
    x0[0] = 1
    x0[4] = 1
    x0[8] = 1
    
    # Solve the minimization, requiring the first rotation matrix to be the identity matrix
    # TODO: figure out (if possible) a better initial guess
    
    res = scipy.optimize.minimize(lambda (x) np.matmult(A, x) - b, x0,
                            constraints=(scipy.minimize.LinearConstraint(constraint_array, bound, bound)))
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
    R_images = np.sum(boardRotMats, axis = 3).transpose() / ncams
    t_images = np.sum(boardTransVecs, axis = 2) / ncams
    
    for i in range(R_images.shape[2]):
        log.VLOG(3, 'Average rotation matrix for Image %d = \n %s' % (i, R_vals[:, :, i]))
    for i in range(t_images.shape[1]):
        log.VLOG(3, 'Average translation vector for Image %d = \n %s' % (i, R_vals[:, :, i]))
    
    

def multiCamCalib1(umeas, xworld, camMatrix, boardRotMat, boardTransVec, planeData, cameraData):

    # set up implementation variables
    nX = planeData.nX
    nY = planeData.nY
    nplanes = planeData.ncalplanes
    ncams = camreaData.ncams

    # set up storage variables
    Rpq = np.zeroes([3, 3, ncams, ncams-1]) # this way [:,:, 0, 1] holds R^(0,1)
    tpq = np.zeroes([3, 1, ncams, ncams-1])

    # loop through all possible pairs of cameras
    for p in range(0, ncams):
        for q in range(p+1, ncams):
            # set up P matrix, which holds all image points in camera p
            pX = umeas[0, :, p]
            pY = umeas[1, :, p]
            pZ = np.zeroes(nX*nY*nplanes)
            P = np.concatenate((pX,pY,pZ)).reshape((-1,3),order='F') # stacks as column vectors in format [x y z]

            # set up Q matrix, which holds all image points in camera q
            qX = umeas[0, :, q]
            qY = umeas[1, :, q]
            qZ = np.zeroes(nX*nY*nplanes)
            Q = np.concatenate((qX,qY,qZ)).reshape((-1,3),order='F')

            # compute matrix H as product of P^T and Q
            H = np.matmul(np.transpose(P), Q)

            # perform svd on matrix H 
            U, S, VTrans = np.linalg.svd(H, full_matrices=True) 

            # calculates variable to correct rotation matrix if necessary
            UTrans = np.transpose(U)
            V = np.transpose(VTrans)
            d = np.det(np.matmul(V, UTrans))

            # calculates optimal rotation matrix R^(p, q)
            I = np.eye(3)
            I[2,2] = d
            tempRpq = np.matmul(np.matmul(V, I), UTrans) 

            # having Rpq, use rigid body motion eq to get t^(p,q)
            temptpq = Q[0,:] - np.matmul(tempRpq, P[0,:])

            # store in storage variables
            Rpq[:,:,p,q] = tempRpq
            tpq[:,:,p,q] = temptpq

    # now that we have all R^(p,q) and t^(p,q), we can setup to solve linear least squares problem

    ######## camera translation ######## 
    # create empty python list A and b that we will append to 
    listA = []
    for i in range(ncams):
        listA.append([None for x in range(ncams)])
    listb = listA.copy() # creates shallow copy

    # loop through our camera pairings, filling up A and b lists in order (q, p)
    for q in range(ncams):
        for p in range(ncams):
            matA = np.zeroes([3,3*ncams])
            matb = np.zeroes([3,1])
            tempRpq = Rpq[:,:, p, q]
            temptpq = tpq[:, :, p, q]

            matA[:, 3*(q-1):3*q] = -tempRpq
            matA[:, 3*(p-1):3*p] = np.eye(3)
            matb = temptpq

            # append these to listA and listb
            listA.append(matA)
            listb.append(matb)

    # convert A list into a matrix 
    A = listA[0]
    listA = listA[1:]
    for e in listA:
        A = np.append(A, e, axis=1)

    # convert b list into a matrix 
    b = listb[0]
    listb = listb[1:]
    for e in listb:
        b = np.append(b, e, axis=1)

    # TODO: FIGURE OUT HOW TO DO THIS
    # define constraint t^1 = 0
    C = np.eye(3, np.size(A, 1)) #B: this could be 2 but matlab indexes at 1 so we will see
    d = np.zeroes([3,1])
    cons1 = lambda x: sum(d - C*x)
    cons = { type:"eq", fun:cons1}

    # solve linear least-squares problem with no bounds and defined constraints
    fun1 = lambda x: 0.5* ??????
    res = scipy.optimize.minimize(fun1, x0, args=(), method='SLSQP',jac=None,hess=None,hessp=None, bounds=None,constraints=cons) 

    # save all translation vectors in variable tVec
    tVec = res.x

    ######### camera rotation ######### 
    # create empty python list A that we will append to
    listA = []
    for i in range(ncams):
        listA.append([None for x in range(ncams)])

    # loop through camera pairings, filling up A list in order (q, p)
    for q in range(ncams):
        for p in range(ncams):
            matA = np.zeroes([9, 9*ncams])
            matA[:, 9*(q-1):9*q] = np.eye(9)

            tempRqp = Rpq[:,:, p, q]

            matA[:, 9*(p-1):9*p] = -np.kron(np.eye(3), tempRqp)

    # convert A list into a matrix 
    A = listA[0]
    listA = listA[1:]
    for e in listA:
        A = np.append(A, e, axis=1)

    # create matrix b 
    b = np.zeroes([size(A,0), 1)] #B: another axis that might be 1 instead of 0

    # define constraints that fix global frame to first view
    C2 = np.eye([9, 9*ncams])
    d2 = np.reshape(np.eye(3), (-1, 1))
    cons2 = lambda x: sum(d2 - C2*x)
    cons = { type:"eq", fun:cons2}

    # define fun and solve linear least-squares
    fun2 = lambda x: 0.5 * np.norm(sum(A*x - b)) # this does Frobenius norm; do I have to square it now?
    res = scipy.optimize.minimize(fun2, x0, args=(), method='SLSQP',jac=None,hess=None,hessp=None, bounds=None,constraints=cons) 

    # save all rotations in variable Rmat
    Rmat = res.x

    ######### checkerboard world positions ######## 
    # for each calib image, estimate position R^n and t^n in world coordinate system

    # set up storage variables
    #avgBoardRotMat = np.zeroes([3, 3*ncalplanes, ncams]) # so avgBoardRotMat[:,0:3,:] has all the images checkerboard placement 1
    #avgBoardTransVec = np.zeroes([3, ncalplanes, ncams])
    avgBoardRotMat = np.zeroes([3, 3*ncalplanes]) # so avgBoardRotMat[:,0:3] has all the images checkerboard placement 1
    avgBoardTransVec = np.zeroes([3, ncalplanes])

    # iterate through each plane in each camera, effectively hitting all images
    for n in range(0, ncalplanes):
        tempRotAvg = boardRotMat[:, 3*n:3*(n+1), 0]
        tempTransAvg = boardTransVec[:, n:(n+1), 0]

        for c in range(0, ncams):
            # add up rotational matrices and board vectors
            tempRotMat = boardRotMat[:, 3*n:3*(n+1), c]
            tempTransVec = boardTransVec[:, 3*n:3*(n+1), c]

            tempRotAvg = np.add(tempRotAvg, tempRotMat)
            tempTransAvg = np.add(tempTransAvg, tempTransVec)

        # divide by total number of cameras ncams to get average 
        tempRotAvg = tempRotAvg/ncams
        tempTransAvg = tempTransAvg/ncams

        # place into storage variables
        avgBoardRotMat[:, 3*n:3*n+1] = tempRotAvg
        avgBoardTransVec[:,n:(n+1)] = tempTransAvg


    ########## reprojection error ########## 

    for n in range(0, ncalplanes):
        for c in range(0, ncams):

    # find the average size of each grid in pixels
    return []

############################################# MAIN LOOP #######################################################

if __name__ == "main_":
    # read config file flag passed from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='relative path to config file', type=str)
    parser.add_argument('-v', '--verbosity', help='verbosity level for file prints', type=int)
    args = parser.parse_args()
    configPath = args.config_file
    log = logger.getLogger(__file__, args.verbosity)

    # read and parse config file
    planeData, cameraData, sceneData, toleranes, calImgs, exptPath, camIDs = parseConfigFile(configPath)

    # find corners and pix_phys
    umeas = findCorners(planeData, cameraData.ncams, exptPath, imgs = calImgs)) 
    # find pixel scaling using passed in parameters
    pix_phys = getScale(planeData, Umeas)

    # perform single camera calibration to get initial calibration matrices 
    # and board geometric changes
    camMatrix, boardRotMat, boardTransVec = singleCamCalib(umeas, xworld, planeData, cameraData)

    # perform multi camera calibration
    # TODO: check inputs after multiCamCalib is fully written
    # TODO: will need more varibale than just x
    x = multiCamCalib(umeas, camMatrix, boardRotMat, boardTransVec, planeData, cameraData)

    # TODO: Change saved data according to waht multiCamCalib returns (we should probably try to make it return these though)
    f = saveCalibData(exptPath, camIDs, P, camParams, Xworld, planeParams, sceneData,cameraData, planeData, errorLog, pix_phys, 'results_')
    print('\nData saved in '+str(f))
