import sys
NUMPAD_DIR = '/home/voila/Documents/2014GRAD/'
sys.path.append(NUMPAD_DIR)
from numpad import *
import numpy as np
import unittest
import nlopt
import matplotlib.pyplot as plt
from pdb import set_trace
import unittest

"Unit test"

class TestMain(unittest.TestCase):
    def setUp(self):
        self.xs = np.linspace(0,1,100)
        self.uinit = np.sin(self.xs)/2. + .5
        self.tinit = 0.
        self.tfinal = .2
        self.source = array([.1, -.1])
        self.tgrid = linspace(self.tinit, self.tfinal, 100)
        # primal parameters
        self.A = 2.
        # twin parameters
        self.nsig = 10
        self.rangesig = [-.1, 1.1]
  
    @unittest.skipIf(False, '')
    def test_primal(self):
        primal = \
        PrimalModel(self.uinit, self.xs, self.tinit, self.tfinal, self.A)
        primal.set_source(self.source)
        primal.integrate(self.tfinal)
        plt.clf()
        primal.plot_utx(self.tgrid)

    @unittest.skipIf(True, '')
    def test_twin(self):
        twin = \
        TwinModel(self.uinit, self.xs, self.tinit, self.tfinal, self.nsig, self.rangesig)
        twin.set_source(self.source)
        twin.flux.setcoef(np.loadtxt('xcoef'))
        twin.integrate(self.tfinal)
        plt.clf()
        twin.plot_utx(self.tgrid)

"Utilities"

def degrade(_adarray_):
    if isinstance(_adarray_, adarray):
        return _adarray_._value
    return _adarray_

def upgrade(_ndarray_):
    if isinstance(_ndarray_, np.ndarray):
        return array(_ndarray_)
    return _ndarray_
        

'Buckley-Leverett flux'

class BLFlux:

    def __init__(self, A):
        self.A = A

    def fluxfun(self, us):
        A = self.A
        fvar = us**2 / (1.+A*(1-us)**2)
        return fvar

    def fluxder(self, us):
        A = self.A
        fder = ( 2*us*(1+A*(1-us)**2) + us**2 * (2*A*(1-us)) ) \
               / (1+A*(1-us)**2)**2
        return fder

    def plotflux(self, cl='r', grad=False):
        x = np.linspace(0.,1.,100)
        if not grad:
            y = self.fluxfun(x)
        else:
            y = self.fluxder(x)
        handle, = plt.plot(x,y,color=cl)
        return handle


'Sigmoid basis library for the flux'

class Flux:

    def __init__(self, nsig, rangesig):
        self.nsig = nsig
        self.beta = 3./2 * nsig
        self.uis = np.linspace(rangesig[0], rangesig[1], nsig)
        self.coef = None
        self.activelist = np.ones(self.uis.shape)

    def activate(self, list_to_activate):
        # activate a list of basis
        list_to_activate = degrade(list_to_activate)
        self.activelist = list_to_activate

    def setcoef(self, coef):
        # set sigmoids coefficients
        assert(coef.size)
        self.coef = upgrade(coef)

    def fluxfun(self, us):
        # evaluate flux function value
        assert(self.coef is not None)
        result = zeros(us.shape)
        for basis in range(self.nsig):
            if bool(self.activelist[basis]):
                result += sigmoid(self.beta* (us - self.uis[basis])) \
                       * self.coef[basis]
        return result

    def fluxder(self, us):
        # compute flux function derivative to u
        assert(self.coef is not None)
        result = zeros(array(us).shape)
        for basis in range(self.nsig):
            if bool(self.activelist[basis]):
                result += sigmoid_der(self.beta * (us - self.uis[basis])) \
                       * self.coef[basis] * self.beta
        return result

    def plotflux(self, cl='b', grad=False):
        distance = self.uis[-1] - self.uis[0]
        lend = self.uis[0]  - .1 * distance
        rend = self.uis[-1] + .1 * distance
        us = linspace(lend._value, rend._value, 1000)
        if not grad:
            y = self.fluxfun(us)._value
        else:
            y = self.fluxder(us)._value
        handle, = plt.plot(us._value, y, color=cl)
        return handle


'Model base class'

class Model:

    def __init__(self, uinit, xs, tinit, tfinal):
        assert( xs.size == uinit.size and isinstance(xs, np.ndarray) )
        self.uinit = uinit
        self.tinit = tinit
        self.tfinal = tfinal
        self.N = uinit.size
        self.xs = xs
        self.dx = self.xs[1] - self.xs[0]
        self.source = None
        self.flux = None
        self.utx = uinit[np.newaxis,:]
        self.ts = np.array(tinit)

    def set_source(self, source):
        # set space dependent design (source)
        # source is constant in time, modelled by bubble profiles in space
        source = upgrade(source)
        dim = source.size
        location = np.linspace(0,1,dim)
        distance = location[1] - location[0]
        profiles = \
        [exp( -(self.xs-center)**2/ distance**2 ) for center in location]
        self.source = sum( [profiles[ii] * source[ii] for ii in range(dim)], 0 )

    def residual(self, un, u0, dt):
        # evolve for one time step
        assert(self.flux is not None)
        un_ext = hstack([un[-2:], un, un[:2]])               # N+4
        fn = self.flux.fluxfun(un_ext)                       # N+4
        lamn = sqrt( self.flux.fluxder(un_ext) ** 2 + 1e-14) # N+4
        coefn = sigmoid( (lamn[:-1] - lamn[1:]) / 1e-6 )     # N+3
        lamn = coefn*lamn[:-1] + (1-coefn)*lamn[1:]          # N+3

        Dn = un_ext[:-1] - un_ext[1:]                        # N+3
        x1n = Dn[:-2]				             # N+1
        x2n = Dn[2:]                                         # N+1

        L = zeros(array(x1n).shape)
        index = (x1n._value * x2n._value > 0.)
        L[ ~ index ] = zeros(np.sum(~index)._value)
        L[ index ] = 2 * (x1n * x2n)[index] / (x1n + x2n)[index]

        fluxn = (fn[1:-2] + fn[2:-1])/2. \
              + .5 * lamn[1:-1] * (Dn[1:-1] - L)
        # -------------------------------------------
        u0_ext = hstack([u0[-2:], u0, u0[:2]])               
        f0 = self.flux.fluxfun(u0_ext)                       
        lam0 = sqrt( self.flux.fluxder(u0_ext) ** 2 + 1e-14) 
        coef0 = sigmoid( (lam0[:-1] - lam0[1:]) / 1e-6 )     
        lam0 = coef0*lam0[:-1] + (1-coef0)*lam0[1:]          

        D0 = u0_ext[:-1] - u0_ext[1:]                        
        x10 = D0[:-2]				         
        x20 = D0[2:]                                         

        L = zeros(array(x10).shape)
        index = (x10._value * x20._value > 0.)
        L[ ~ index ] = zeros(np.sum(~index)._value)
        L[ index ] = 2 * (x10 * x20)[index] / (x10 + x20)[index]

        flux0 = (f0[1:-2] + f0[2:-1])/2. \
              + .5 * lam0[1:-1] * (D0[1:-1] - L)
        # -------------------------------------------
        if self.source is None:
            print 'warning: source unset'
        res = (un - u0)/dt + (fluxn[1::]-fluxn[:-1:])/self.dx/2.\
            + (flux0[1::]-flux0[:-1:])/self.dx/2. - self.source 
        return res

    def integrate(self, tcutoff):
        self.ts = np.array([self.tinit])
        tnow = self.tinit
        dt = (self.tfinal - self.tinit)/1e3
        endt = np.min([self.tfinal, tcutoff])

        while tnow<endt:
            adsol = solve(self.residual, self.utx[-1], \
                          args = (self.utx[-1], dt), \
                          max_iter=100, verbose=False)
            tnow += dt
            self.utx = vstack([self.utx, adsol.reshape([1,adsol.size])])
            self.ts = hstack([self.ts, np.array(tnow)])
            if adsol._n_Newton < 4:
                dt *= 2.
            elif adsol._n_Newton > 16:
                dt /= 2.

    def interp_tgrid(self, tgrid):
        # interp utx from ts to tgrid
        utx_grid = zeros([tgrid.size, self.N])
        for ix in range(self.N):
            interp_base = interp(self.ts, self.utx[:,ix])
            utx_grid[:,ix] = interp_base(tgrid)
        return utx_grid

    def plot_utx(self, tgrid):
        utx_grid = self.interp_tgrid(tgrid)      
        T,X = np.meshgrid(degrade(tgrid), degrade(self.xs))
        plt.contourf(T,X,degrade(utx_grid))


'Primal model'

class PrimalModel(Model):
    
    def __init__(self, uinit, xs, tinit, tfinal, A):
        Model.__init__(self, uinit, xs, tinit, tfinal)
        self.flux = BLFlux(A)
 
'Twin model'
   
class TwinModel(Model):

    def __init__(self, uinit, xs, tinit, tfinal, nsig, rangesig):
        Model.__init__(self, uinit, xs, tinit, tfinal)
        self.flux = Flux(nsig, rangesig)


'Infer twin model'

class InferTwinModel:

    def __init__(self):
        pass

    def mismatch(self):
    # solution mismatch, with Lasso basis selection
    # map twin model solution to primal model's time grid
        pass

    def infer(self):
    # optimize selected basis coefficients
        pass


'Bayesian optimization'
class BayesOpt:

    def __init__(self):
        pass

    def likelihood(self):
        pass

    def mle(self):
    # maximum likelihood estimate of hyperparameters
        pass
    
    def posterior(self):
    # posterior evaluation
        pass

    def next_design(self):
    # next candidate design
        pass

    def optimize(self):
    # optimize space-time dependent design
        pass

if __name__ == '__main__':
    unittest.main()
