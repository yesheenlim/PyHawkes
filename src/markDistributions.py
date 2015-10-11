
import random
import math
import scipy.optimize as op

class ParetoPolynomial:
    def __init__(self):
        """
        Pareto distribution (as defined by Liniger) with polynomial
        normalized impact function, totalling 5 parameters. mu and rho
        are assumed to be estimated seperately from the  model.

        distribution param: [mu, rho]
        impact param: [alpha, beta, gamma]
        """
        self.mu,self.rho,self.alpha,self.beta,self.gamma = (1e-5,2.0+1e-5,1e-5,0,0)

    def setDistParam(self, param):
        self.mu,self.rho = param

    def setImpactParam(self, param):
        self.alpha,self.beta,self.gamma = param

    @staticmethod
    def getNumDistParam():
        return 2

    @staticmethod
    def getNumImpactParam():
        return 3

    @staticmethod
    def getDistBounds():
        ub = None
        return [(1e-5,ub),(2.0+1e-5,ub)]

    @staticmethod
    def getImpactBounds():
        ub = None
        return [(1e-5,ub),(0,ub),(0,ub)]

    def sample(self):
        randomVal = random.random()
        return self._inverseCumulativeDistribution(randomVal)

    def _inverseCumulativeDistribution(self,x):
        c = math.pow((1.0-x),(1.0/self.rho)) - 1.0
        d = math.pow((1.0-x),(-1.0/self.rho))
        retval = -self.mu*c*d
        return retval

    def Density(self,x):
        density = self.rho*(math.pow(self.mu,self.rho))/(math.pow((x+self.mu),(self.rho+1)))
        return density

    def CumulativeDistribution(self,x):
        return 1.0 - math.pow((self.mu/(x+self.mu)),self.rho)

    def Impact(self,x):
        numerator = (self.rho-1.0)*(self.rho-2.0)*(self.alpha+self.beta*x+self.gamma*x*x)
        denominator = self.alpha*(self.rho-1.0)*(self.rho-2.0)+self.beta*self.mu*(self.rho-2.0)+2.0*self.gamma*self.mu*self.mu
        return numerator/denominator

class ParetoLinear:
    def __init__(self):
        """
        Pareto distribution (as defined by Liniger) with linear
        normalized impact function, totalling 4 parameters. mu and rho
        are assumed to be estimated seperately from the model.

        distribution param: [mu, rho]
        impact param: [alpha, beta]
        """
        self.mu,self.rho,self.alpha,self.beta = (1e-5,2.0+1e-5,1e-5,0)

    def setDistParam(self, param):
        self.mu,self.rho = param

    def setImpactParam(self, param):
        self.alpha,self.beta = param

    @staticmethod
    def getNumDistParam():
        return 2

    @staticmethod
    def getNumImpactParam():
        return 2

    @staticmethod
    def getDistBounds():
        ub = None
        return [(1e-5,ub),(2.0+1e-5,ub)]

    @staticmethod
    def getImpactBounds():
        ub = None
        return [(1e-5,ub),(0,ub)]

    def sample(self):
        randomVal = random.random()
        return self._inverseCumulativeDistribution(randomVal)

    def _inverseCumulativeDistribution(self,x):
        c = math.pow((1.0-x),(1.0/self.rho)) - 1.0
        d = math.pow((1.0-x),(-1.0/self.rho))
        retval = -self.mu*c*d
        return retval

    def Density(self,x):
        density = self.rho*(math.pow(self.mu,self.rho))/(math.pow((x+self.mu),(self.rho+1)))
        return density

    def CumulativeDistribution(self,x):
        return 1.0 - math.pow((self.mu/(x+self.mu)),self.rho)

    def Impact(self,x):
        numerator = (self.rho-1.0)*(self.rho-2.0)*(self.alpha+self.beta*x)
        denominator = self.alpha*(self.rho-1.0)*(self.rho-2.0)+self.beta*self.mu*(self.rho-2.0)
        return numerator/denominator

    def LogLikelihood(self, theta, *args):
        self.mu, self.rho = theta
        data = args

        loglikelihood = 0
        for x in data:
            density = self.Density(x)
            loglikelihood += math.log(density)

        return loglikelihood

    def MLE(self,x,method=None,x0=None):
        if x0 == None:
            x0 = self._initRandomValues()
        nLL = lambda *args: - self.LogLikelihood(*args)
        result = op.minimize( fun = nLL,
                                    x0 = x0,
                                    args = x,
                                    method = method,
                                    jac = False,
                                    bounds = self.getDistBounds())
        return result


    def _initRandomValues(self):
        rand = []
        for lb,ub in self.getDistBounds():
            if lb is None:
                lb = 0.0
            rand.append(random.uniform(lb,lb+1.0))
        return rand
