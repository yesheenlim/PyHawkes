
import numpy as np

class ImmigrationBranchingParameters:
    def __init__(self,numComponents):
        '''
        Data structure to store immigration and branching parameters.
        '''
        self.numComponents = numComponents
        self.eta = np.empty(numComponents); self.eta.fill(1e-5)
        self.Q = np.empty([numComponents,numComponents]); self.Q.fill(1e-5)

    def getNumParam(self):
        return (self.numComponents*self.numComponents)+self.numComponents

    def getParamBounds(self):
        ub = 30
        return [(1e-5,ub)]*self.getNumParam()

    def setParam(self,eta,Q):
        for idx,val in zip(xrange(self.numComponents),eta):
            self.eta[idx] = val
        for idx,val in zip(xrange(self.numComponents*self.numComponents),Q):
            self.Q.put(idx,val)

    def getSpectralRadius(self):
        eigenVal,eigenVect = np.linalg.eig(self.Q)
        return np.amax(abs(eigenVal))
