import numpy as np
import numpy.linalg as lin
import cv2
import glob

def getImgs(path, n):
    
    img = []
    for i in range(1,10):
        cpath = path + 'cam%d' % i
        imgs = glob.glob(cpath+'/*')
        I = cv2.imread(imgs[n], 0)
        img.append(I)
    
    return(img)

def backProj(p, imx, imy, z):

    h = p[:,0:3]
    h[:,2] = h[:,2]*z + p[:,3]
    h = lin.inv(h)
    scale = 10
    d = np.matrix([[scale,0,imx*0.5],[0,scale,imy*0.5],[0,0,1]])
    h = d*h

    return(h)

def cameraMat(c, imx, imy, f, ss):
    
    tx = np.arcsin(c[0]/c[2])
    ty = -np.arcsin(c[1]/c[2])
    tz = 0
    
    R = rotMat(tx,ty,tz)
    
    scale = f*imx/ss
    k = np.matrix([[scale,0,imx*0.5],[0,scale,imy*0.5],[0,0,1]])
    
    Rt = np.matrix(np.zeros((3,4)))
    Rt[:,0:3] = R
    Rt[:,3] = -R*np.transpose(np.matrix(c))
    Rt = np.matrix(Rt)
    
    P = np.copy(k*Rt)
    
    return(P)

def rotMat(tx, ty, tz):

    Rx = np.matrix([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.matrix([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.matrix([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
    R = np.matrix(Rx*Ry*Rz);
    
    return(R)
