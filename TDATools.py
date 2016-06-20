import numpy as np
import matplotlib.pyplot as plt

class UFNode(object):
    def __init__(self, birthTime):
        self.P = self
        self.birthTime = birthTime
        self.deathTime = np.inf

#Union find "find" with path compression
def UFFind(u):
    if not u.P == u:
        u.P = UFFind(u.P) #Path compression
        return u.P
    return u

#Union find "union" with merging to component with earlier birth time
def UFUnion(u, v, time):
    uP = UFFind(u)
    vP = UFFind(v)
    if uP == vP:
        return #Already in union
    #Merge to the root of the one with the earlier component time
    [ufirst, usecond] = [uP, vP]
    if usecond.birthTime < ufirst.birthTime:
        [usecond, ufirst] = [ufirst, usecond]
    usecond.P = ufirst
    usecond.deathTime = time

#X: Time Series
#Return Nx3 array of birth/death/index pairs, where N is the number
#of nonzero classes
def getMorseFiltration(X):
    N = len(X)
    arr = [UFNode(X[i]) for i in range(N)]
    idxs = np.argsort(X)
    for i in idxs:
        if i-1 >= 0:
            UFUnion(arr[i-1], arr[i], X[i])
        if i+1 < N:
            UFUnion(arr[i+1], arr[i], X[i])
    I = []
    for i in range(len(arr)):
        node = arr[i]
        if node.deathTime > node.birthTime:
            if np.isinf(node.deathTime):
                I.append([node.birthTime, np.max(X), i])
            else:
                I.append([node.birthTime, node.deathTime, i])
    return np.array(I)

if __name__ == '__main__':
    t = np.linspace(0, 2, 1000)
    x = -np.exp(-(t-0.5)**2/0.1)
    x -= 2*np.exp(-(t-1.5)**2/0.1)
    x = x - np.min(x) + 1 + 0.01*np.random.randn(len(x))
    I = getMorseFiltration(x)
    idxs = np.array(I[:, 2], dtype=np.int64)
    idxs = idxs[(I[:, 1] - I[:, 0]) > 0.1]
    print "idxs = ", idxs
    
    plt.subplot(121)
    plt.plot(x, 'b')
    plt.hold(True)
    plt.plot(idxs, x[idxs], 'rx')
    
    plt.subplot(122)
    plt.plot(I[:, 0], I[:, 1], 'b.')
    plt.hold(True)
    plt.plot([0, np.max(I[:, 0:2])], [0, np.max(I[:, 0:2])], 'r')
    plt.show()
    
