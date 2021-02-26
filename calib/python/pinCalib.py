# Pseudocode/General Overview:
# parse config file, get calibration images, get camera IDs, perform variable setups
# 
# find chessboard corners in all images and store them in variable called cor_all
#
# create "real world" coordinates using grid spacing and number of corners
#
# line up real world coordinates with corner of checkerboard image
#       use planar_grid to world
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

if __name__ == "main_":

    # read and parse config file
    planeData, cameraData, sceneData, toleranes, calImgs, exptPath, camIDs = parseConfigFile()

    # find corners and pix_phys
    umeas = findCorners( ... ) 

    # perform single camera calibration to get initial calibration matrices 
    # and board geometric changes
    camMatrix, boardRotMat, boardTransVec = singleCamCalib(umeas, xworld, planeData, cameraData)

    # perform multi camera calibration
    x = multiCamCalib(umeas, camMatrix, boardRotMat, boardTransVec, planeData, cameraData)

def multiCamCalib(umeas, camMatrix. boardRotMat, boardTransVec, planeData, cameraData):

    # set up implementation variables
    nX = planeData.nX
    nY = planeData.nY
    nplanes = planeData.ncalplanes
    ncams = camreaData.ncams
    Umeas = umeas

    # set up storage variables
    Rpq = np.zeroes([3, 3, ncams, ncams-1]) # this way [:,:, 0, 1] holds R^(0,1)
    tpq = np.zeroes([3, 1, ncams, ncams-1])

    # loop through all possible pairs of cameras
    for p in range(0, ncams):
        for q in range(p+1, ncams):
            # set up P matrix, which holds all image points in camera p
            pX = umeas[0, :, p]
            pY = umeas[1,:,p]
            pZ = np.zeroes(nX*nY*nplanes)
            P = np.concatenate((pX,pY,pZ)).reshape((-1,3),order='F') # stacks as column vectors in format [x y z]

            # set up Q matrix, which holds all image points in camera q
            qX = umeas[0, :, q]
            qY = umeas[1,:,q]
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

    # TODO: FIGURE OUT HOW TO DO THE MINIMIZATION lmao
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

def singleCamCalib(umeas, xworld, planeData, cameraData):
    #Inputs:
    # umeas is 2 x N x ncams array of image points in each camera
    # xworld is 3 x nX*nY*nplanes of all world points
    #
    #Outputs:
    # camMatrix     - 3 x 3 x ncams that holds all camera calibration matrices
    # boardRotMat   - 3 x 3*nplanes x ncams for each board in each camera
    # boardTransVec - 3 x nplanes x ncams for each board in each camera

    # set up implementation variables
    nX = planeData.nX
    nY = planeData.nY
    nplanes = planeData.ncalplanes
    ncams = cameraData.ncams
    imageSize = size(cameraData.sX, cameraData.sY)
    aspectRatio = cameraData.sX / cameraData.sY

    # set up storage variables 
    cameraMatrix = np.zeroes([3, 3, ncams])
    boardRotMat = np.zeroes([3, 3*nplanes, ncams])
    boardTransVec = np.zeroes([3, nplanes, ncams])

    # cycle through each camera 
    for i in range(0, ncams):

        # calculate initial camera matrix for this camera 
        # use all world points but only image points in this camera
        # calls opencv function initCameraMatrix
        camUmeas = umeas[:, :, i]
        camMatrix = initCameraMatrix2D(xworld, camUmeas, nX*nY, imageSize, aspectRatio)

        # iterate through each board placement to find rotation matrix and rotation vector
        for n in range(0, ncalplanes):

            # isolate only points related to this board's placement
            currWorld = xworld[:, nX*nY*n:nX*nY*(n+1)]
            currImage = umeas[:, nX*nY*n:nX*nY*(n+1), i]

            # find the board's rotation and translation vector
            _, rotMatrix, transVector = solvePnP(currWorld, currImage, camMatrix, np.zeroes((8,1), dtype='float32')) 

            # add board values to storage variables
            boardRotMat[:, 3*n:3*(n+1), i] = rotMatrix
            boardTransVec[:, n:(n+1), i] = transVector 

        # add camera matrix to storage variable
        cameraMatrix[:,:, i] = camMatrix

    return [cameraMatrix, boardRotMat, boardTransVec]


