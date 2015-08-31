
import math
import random
import utils
import numpy as np
import markDistributions as md
import scipy.optimize as op

class Hawkes:
    def __init__(self,d):
        '''
        Implements the Hawkes marked point process as described in
        Liniger's thesis [Liniger, 2012], assuming a polynomial decay
        function.

        d: number of multivariate point process components (scalar, N)
        '''
        self.numComponents = d
        self.ibParam = utils.ImmigrationBranchingParameters(self.numComponents)
        self.alpha = np.empty(self.numComponents); self.alpha.fill(1e-5)
        self.markDist = []
        for i in xrange(self.numComponents):
            self.markDist.append(md.ParetoLinear())

    def setParam(self,param):
        '''
        Sets the parameters for the Hawkes point process model.

        param: (eta, Q, alpha, impactParam)
        - list of length d+(d*d)+d+(d*3)
        '''
        eta = param[:self.numComponents]
        Q = param[len(eta):len(eta)+(self.numComponents*self.numComponents)]
        alpha = param[len(eta)+len(Q):len(eta)+len(Q)+self.numComponents]
        impactParam = []
        for i in xrange(self.numComponents):
            n = self.markDist[i].getNumImpactParam()
            impactParam.append(param[len(eta)+len(Q)+len(alpha)+(n*i):len(eta)+len(Q)+len(alpha)+(n*i)+n])

        self.ibParam.setParam(eta,Q)
        self.alpha = np.asarray(alpha)
        for i,val in zip(xrange(self.numComponents),impactParam):
            self.markDist[i].setImpactParam(val)

    def setMarkDistParam(self,param):
        '''
        Sets the parameters mu and rho for the mark distribution,
        for which the paremeters are esimated seperately from the
        Hawkes model.
        '''
        for i,val in zip(xrange(self.numComponents),param):
            self.markDist[i].setDistParam(val)

    def Intensity(self,j,t,_tjx,_intensity):
        '''
        Computes the intensity estimation of component j at time t.
        '''
        _t,_j,_x = _tjx
        intensity = self.ibParam.eta[j] + math.exp(-self.alpha[j]*(t-_t))*(_intensity-self.ibParam.eta[j]) + \
                    self.ibParam.Q[j,_j]*self.alpha[j]*math.exp(-self.alpha[j]*(t-_t))*self.markDist[_j].Impact(_x)

        return intensity

    def simulate(self,numTimesteps):
        '''
        Simulate a Hawkes marked point process for numTimesteps timesteps.
        Due to the required burn-in period, the first few hunderd points are
        discarded before returned.
        '''
        burn = 200
        time = np.zeros(numTimesteps)
        component = np.zeros(numTimesteps, dtype='int')
        mark = np.zeros(numTimesteps)
        intensities = np.empty([numTimesteps,self.numComponents])
        intensities.fill(1e-5); intensities[0,] = self.ibParam.eta

        for timestep in xrange(1,numTimesteps):
            _tjx = (time[timestep-1],component[timestep-1],mark[timestep-1])
            newTime = float('inf')
            newComponent = 0

            for j in xrange(self.numComponents):
                tau = self._innerLoop(j=j,_tjx=_tjx,_intensity=intensities[timestep-1,j])
                if tau < newTime:
                    newTime = tau
                    newComponent = j

            time[timestep] = newTime
            component[timestep] = newComponent
            mark[timestep] = self.markDist[newComponent].sample()

            for j in xrange(self.numComponents):
                intensities[timestep,j] = self.Intensity(j=j,_tjx=_tjx,_intensity=intensities[timestep-1,j],t=newTime)

        return (time[burn:],component[burn:],mark[burn:]), intensities[burn:,]

    def _innerLoop(self,j,_tjx,_intensity):
        '''
        Executes the inner loop of the thinning process.
        '''
        tau = _tjx[0]
        intensity = self.Intensity(j=j,_tjx=_tjx,_intensity=_intensity,t=tau)
        while True:
            tau += random.expovariate(1.0) / intensity
            intensity_ = self.Intensity(j=j,_tjx=_tjx,_intensity=intensity,t=tau)

            if random.random()*intensity <= intensity_:
                return tau
            else:
                intensity = intensity_

    def LogLikelihood(self,x,*args):
        '''
        Returns the Hawkes log-likelihood for a given observed period.

        *args: the time, component, mark array tuple
        x: the model parameters
        '''
        self.setParam(x)
        _t, _j, _x = args

        _intensity = self.ibParam.eta
        intensity = np.zeros(self.numComponents)

        loglikelihood = math.log(_intensity[_j[0]] * self.markDist[_j[0]].Density(_x[0]))
        for m in xrange(1,len(_t)):
            _tjx = (_t[m-1],_j[m-1],_x[m-1])
            tm,dm,xm = (_t[m],_j[m],_x[m])

            for k in xrange(self.numComponents):
                intensity[k] = self.Intensity(j=k,t=tm,_tjx=_tjx,_intensity=_intensity[k])

            loglikelihood += math.log(intensity[dm] * self.markDist[dm].Density(xm))

            _intensity = intensity

        for ite in xrange(self.numComponents):
            loglikelihood -= self._compensator(ite,args=args)

        return loglikelihood

    def _compensator(self,j,args):
        '''
        Returns the compensator.
        '''
        _t,_j,_x = args
        _T = _t[-1]
        compensator = self.ibParam.eta[j] * (_T - _t[0])

        for m in xrange(len(_t)-1-1,-1,-1):
            tm,dm,xm = (_t[m],_j[m],_x[m])
            if _T - tm > self._quantile(j):
                compensator += self.ibParam.Q[j,dm]*self.markDist[dm].Impact(xm)
            else:
                compensator += self.ibParam.Q[j,dm]*self._cumulativeDecay(j,_T-tm)*self.markDist[dm].Impact(xm)

        return compensator

    def _quantile(self,j):
        '''
        Returns the quantile function value.
        '''
        eps = 1e-5
        return -math.log(eps)/self.alpha[j]

    def _cumulativeDecay(self,j,t):
        '''
        Returns the cumulative decay function value.
        '''
        return 1.0 - math.exp(-self.alpha[j]*t)

    def _paramBounds(self):
        '''
        Aggregate the bounds for eta, Q, alpha, markParam, in that order.
        '''
        bound = []
        bound.extend(self.ibParam.getParamBounds())
        bound.extend([(1e-5,10)]*self.numComponents)
        for i in xrange(self.numComponents):
            bound.extend(self.markDist[i].getImpactBounds())

        return bound

    def _initRandomValues(self):
        '''
        Randomizes initial parameter.
        '''
        rand = []
        for lb,ub in self._paramBounds():
            if lb is None:
                lb = 0.0
            rand.append(random.uniform(lb,lb+1.0))

        return rand

    def MLE(self,t,j,x,method=None,x0=None):
        '''
        Estimates the parameters of the Hawkes model using maximum
        likelihood estimation, via the scipy's optimization library.
        '''
        if x0 == None:
            x0 = self._initRandomValues()

        nLL = lambda *args: - self.LogLikelihood(*args)
        result = op.minimize( fun = nLL,
                            x0 = x0,
                            args = (t,j,x),
                            method = method,
                            jac = False,
                            bounds = self._paramBounds())
        return result

    def Residual(self,t,j,x):
        '''
        Returns the residual process of the original data, given the
        assumption that the original Hawkes model is correct.
        '''
        tau = np.zeros(len(t))
        xi = np.zeros(len(x))
        Theta = np.zeros((len(t),self.numComponents))
        q = max(self._quantile(0),self._quantile(1))

        for i in xrange(1,len(t)):
            for e in xrange(self.numComponents):
                Theta[i,e] = Theta[i-1,e] + self.ibParam.Q[e,j[i]]*self.markDist[j[i]].Impact(x[i])

        mask = []
        for i in xrange(len(t)):
            di = j[i]
            for m0 in xrange(len(t)):
                if t[i]-t[m0] <= q and t[i]-t[m0-1] > q:
                    mask.append(i)
                    tau[i] = self.ibParam.eta[di]*(t[i]-t[0]) + Theta[m0-1,di]
                    for m in xrange(m0,i-1):
                        tm,dm,xm = (t[m],j[m],x[m])
                        tau[i] += self.ibParam.Q[di,dm]*self._cumulativeDecay(di,t[i]-tm)*self.markDist[dm].Impact(xm)
                    xi[i] = self.markDist[di].sample()
                    break

        return (j[mask], t[mask], x[mask], tau[mask], xi[mask])
