import sys
sys.path.append("S3DGLPy")
from Primitives3D import *
from PolyMesh import *
from scipy.special import sph_harm

import scipy.sparse
import scipy.stats
import scipy.sparse.linalg as slinalg
import numpy as np
import matplotlib.pyplot as plt

def getLaplacianEigs(A, NEigs):
    DEG = scipy.sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG - A
    w, v = slinalg.eigsh(L, k=NEigs, sigma = 0, which = 'LM')
    return (w, v)

def getAdjFromMesh(mesh):
    N = len(mesh.vertices)
    I = []
    J = []
    for e in mesh.edges:
        [v1, v2] = [e.v1.ID, e.v2.ID]
        I.append(v1)
        J.append(v2)
        J.append(v1)
        I.append(v2)
    I = np.array(I)
    J = np.array(J)
    A = scipy.sparse.coo_matrix((np.ones(len(I)), (I, J)), shape=(N, N)).tocsr()
    return A

######################################################
##                   Meshes                         ##
######################################################


def getRectMesh(M, N):
    #Create an M x N grid
    idx = np.arange(M*N)
    idx = np.reshape(idx, (M, N))
    [XPos, YPos] = np.meshgrid(np.arange(N), np.arange(M))
    m = PolyMesh()
    for i in range(XPos.size):
        m.addVertex(np.array([XPos.flatten()[i], YPos.flatten()[i], 0]))
    for i in range(M-1):
        for j in range(N-1):
            a = m.vertices[idx[YPos[i, j], XPos[i, j]]]
            b = m.vertices[idx[YPos[i+1, j], XPos[i+1, j]]]
            c = m.vertices[idx[YPos[i+1, j+1], XPos[i+1, j+1]]]
            d = m.vertices[idx[YPos[i, j+1], XPos[i, j+1]]]
            m.addFace([a, b, c])
            m.addFace([a, c, d])
    return m

def getTorusMesh(M, N, R1, R2):
    #Create an M x N grid
    idx = np.arange(M*N)
    idx = np.reshape(idx, (M, N))
    [XPos, YPos] = np.meshgrid(np.arange(N), np.arange(M))
    [theta1, theta2] = np.meshgrid(np.linspace(0, 2*np.pi, N+1)[0:N], np.linspace(0, 2*np.pi, M+1)[0:M])
    X = (R1 + R2*np.cos(theta1))*np.cos(theta2)
    Y = (R1 + R2*np.cos(theta1))*np.sin(theta2)
    Z = R2*np.sin(theta1)
    m = PolyMesh()
    for i in range(XPos.size):
        m.addVertex(np.array([X.flatten()[i], Y.flatten()[i], Z.flatten()[i]]))
    for i in range(M):
        for j in range(N):
            i1 = YPos[i, j]
            i2 = (i1+1)%M
            j1 = XPos[i, j]
            j2 = (j1+1)%N
            a = m.vertices[idx[i1, j1]]
            b = m.vertices[idx[i2, j1]]
            c = m.vertices[idx[i2, j2]]
            d = m.vertices[idx[i1, j2]]
            m.addFace([a, b, c])
            m.addFace([a, c, d])
    return m

#Return spherical harmonics on the sphere mesh.  Assumes a unit sphere
def getSphericalHarmonics(m):
    cosphi = m.VPos[:, 2]
    sinphi = np.sqrt(1-cosphi**2)
    costheta = m.VPos[:, 0]/sinphi
    sintheta = m.VPos[:, 1]/sinphi
    costheta[sinphi == 0] = 1
    sintheta[sinphi == 0] = 1
    cosphi[cosphi < -1] = -1
    cosphi[cosphi > 1] = 1
    costheta[costheta < -1] = -1
    costheta[costheta > 1] = 1
    phi = np.arccos(cosphi)
    theta = np.arccos(costheta)
    Y = sph_harm(1, 2, theta, phi)
    Y = np.real(Y)
    return Y

#Get the adjacency matrix for a circle
def getCircleAdj(NPoints):
    I = np.arange(NPoints).tolist()
    J = np.arange(NPoints)
    J = J + 1
    J[-1] = 0
    J = J.tolist()
    IF = np.array(I + J)
    JF = np.array(J + I)
    A = scipy.sparse.coo_matrix((np.ones(len(IF)), (IF, JF)), shape=(NPoints, NPoints)).tocsr()
    return A

if __name__ == '__main__1':
    NPoints = 1000
    NEigs = 4
    A = getCircleAdj(NPoints)
    (w, v) = getLaplacianEigs(A, NEigs)
    print w
    cmap = plt.get_cmap('jet')
    theta = np.linspace(0, 2*np.pi, NPoints+1)
    theta = theta[0:NPoints]
    x = np.cos(theta)
    y = np.sin(theta)
    for i in range(NEigs):
        thisv = v[:, i]
        thisv = thisv - np.min(thisv)
        thisv = thisv/np.max(thisv)
        C = cmap(thisv)[:, 0:3]
        plt.scatter(x, y, 20, C, edgecolors = 'none')
        plt.show()

if __name__ == '__main__2':
    m = getSphereMesh(1, 6)
    A = getAdjFromMesh(m)
    NEigs = 10
    (w, v) = getLaplacianEigs(A, NEigs)
    cmap = plt.get_cmap('jet')
    for i in range(NEigs):
        thisv = v[:, i]
        thisv = thisv - np.min(thisv)
        thisv = thisv/np.max(thisv)
        m.VColors = 255*cmap(thisv)[:, 0:3]
        m.saveOffFile("sphere%i.off"%i, output255 = True)


if __name__ == '__main__':
    #m = getSphereMesh(1, 4)
    m = getTorusMesh(200, 200, 4, 1)
    A = getAdjFromMesh(m)
    NEigs = 7
    (w, v) = getLaplacianEigs(A, NEigs)
    print w
    cmap = plt.get_cmap('jet')
    for i in range(1, 7):
        for j in range(i+1, 7):
            plt.clf()
            plt.plot(v[:, i], v[:, j], '.')
            plt.savefig("Plot%i_%i.png"%(i, j))
            theta = np.arctan2(v[:, j], v[:, i])
            theta = theta - np.min(theta)
            theta = theta/np.max(theta)
            m.VColors = 255*cmap(theta)[:, 0:3]
            m.saveOffFile("torus%i_%i.off"%(i, j), output255 = True)
    
    for i in range(NEigs):
        thisv = v[:, i]
        thisv = thisv - np.min(thisv)
        thisv = thisv/np.max(thisv)
        m.VColors = 255*cmap(thisv)[:, 0:3]
        m.saveOffFile("torus%i.off"%i, output255 = True)
