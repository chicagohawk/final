import sys
sys.path.append('../..')
from numpad import *
import numpy as np
from pdb import set_trace
import unittest
import nlopt
import matplotlib.pyplot as plt
'''
class TestFluxclass(unittest.TestCase):

    def setUp(self):
        self.flux = fluxclass(3, [-1.,1.])
        self.flux.update_coef(array([0,1,3]))

    def test_fun_der(self):
        xs = linspace(-2,2,100)
        funval = self.flux.fluxfun(xs)
        derval = self.flux.fluxder(xs)
        dxs = linspace(-2,2,100)+1e-6
        dfunval = self.flux.fluxfun(dxs)
        plt.plot(xs._base, funval._base)
        plt.plot(xs._base, derval._base)
        plt.plot(xs._base, (dfunval-funval)._base/1e-6)
'''

class TestSolve(unittest.TestCase):
    
    def setUp(self):
        # use its initial condition
        uref = loadtxt('sol/solution2.txt') 
        tfinal = .1
        tref = linspace(0., tfinal, uref.shape[0])
        self.twin = twinmodel(uref, tref, tfinal)
        # self.twin.update_flux(
        #    np.random.random(self.twin.flux.ns))
        self.twin.update_flux(loadtxt('xcoef'))

    def test_tintegral(self):
        controls = linspace(-2.,2.,15)
        Js = zeros(controls.shape)
        for ii in range(15):
            print ii
            self.twin.force = controls[ii]
            self.twin.tintegral()
            Js[ii] = self.twin.objective(\
                     loadtxt('sol/solution0.txt'))
            set_trace()

'''
class TestMultiModel(unittest.TestCase):

    def setUp(self):
        self.mtwin = multimodels(nsol=2)

    def test_fun_grad(self):
        coef = np.random.random(self.mtwin.models[0].flux.ns)
        grad = np.array([])
        mJ = self.mtwin.func_grad(coef, grad)
        coef += np.ones(self.mtwin.models[0].flux.ns)*1e-5
        grad = np.zeros(coef.shape)
        dmJ = self.mtwin.func_grad(coef, grad)
'''

class multimodels:

    def __init__(self, nsol=1):
        self.nsol = nsol
        # urefs = [loadtxt('sol/solution%d.txt' % i)\
        #          for i in range(nsol)]
        urefs = [loadtxt('sol/solution2.txt')]
        self.models = [twinmodel(urefs[ii], linspace(\
                       0., .1, urefs[ii].shape[0]), .1)\
                       for ii in range(nsol)]
    
    def func_grad(self, coef, grad):
        Js = zeros(self.nsol)
        for i in range(self.nsol):
            self.models[i].update_flux(coef)
            Js[i] = self.models[i].objective()
        if grad.size>0:
            for icoef in range(self.models[0].flux.ns):
                grad[icoef] = 0.
            for i in range(self.nsol):
                igrad = Js[i].diff(self.models[i].flux.coef)
                if isinstance(igrad, int):
                    for icoef in range(grad.size):
                        grad[icoef] += 0.
                else:
                    for icoef in range(grad.size):
                        grad[icoef] += np.array(igrad)[0][icoef]
            print 'objective func: ' + str(sum(Js._base))
        return sum(Js)._base.flatten()[0]
            
        

class fluxclass:

    def __init__(self, ns, urange):
        self.ns = ns
        self.uis = linspace(urange[0], urange[1], ns)
        self.beta = 3./2 * ns
        self.coef = None

    def update_coef(self, coef, verbose=True):
        assert(coef.size)
        self.coef = array(coef)
        if verbose is True:
            print 'flux coef updated: ' + str(coef)

    def fluxfun(self, us):
        if self.coef is None:
            print 'fatal error: coef is none'
            exit()
        result = zeros(array(us).shape)
        for basis in range(self.ns):
            result += sigmoid(self.beta* (us - self.uis[basis])) \
                      * self.coef[basis]
        return result

    def fluxder(self, us):
        '''
        derivative to u
        '''
        assert(self.coef is not None)
        result = zeros(array(us).shape)
        for basis in range(self.ns):
            result += sigmoid_der(self.beta * (us - self.uis[basis]))\
                      * self.coef[basis] * self.beta
        return result

    def plotflux(self):
        us = linspace(0.,1.,100)
        plt.figure()
        plt.plot(us._base, self.fluxfun(us)._base)
        

class twinmodel:

    def __init__(self, uref, tref, tfinal, ns=10, urange=[-0.2,1.2]):
        self.uref = uref.copy()
        self.tref = tref.copy()
        self.N = uref.shape[1]
        self.ts = tref[0].reshape([1,])
        self.tfinal = tfinal
        self.xs = linspace(0, 1, self.N)
        self.dx = self.xs[1] - self.xs[0]
        self.utx = uref[0].reshape([1,self.N])
        
        self.flux = fluxclass(ns, urange)
        self.J = None
        self.force = None

    def update_flux(self, coef):
        self.flux.update_coef(coef)
        self.J = None
        self.utx = self.uref[0].reshape([1,self.N])
        self.ts = self.tref[0].reshape([1,])

    def residual(self, un, u0, dt, force=None, scheme='BE'):
        '''
        res = (un - u0)/dt + dF(un)/dx - force
        '''
        if force is None:
            force = zeros(u0.shape)
        if scheme is 'BE':
            un_ext = hstack([un[0], un, un[-1]])
            fn_ext = self.flux.fluxfun(un_ext)
            g2n_ext = self.flux.fluxder(un_ext)**2
            limiter = sqrt(g2n_ext[:-1:] + g2n_ext[1::])
            fluxn = (fn_ext[:-1:] + fn_ext[1::])/2. \
                  - limiter * (un_ext[1::] - un_ext[:-1:])/2.
            res = (un - u0)/dt + (fluxn[1::]-fluxn[:-1:])/self.dx\
                - force
        elif scheme is 'Trapozoidal':
            un_ext = hstack([un[0], un, un[-1]])
            fn_ext = self.flux.fluxfun(un_ext)
            g2n_ext = self.flux.fluxder(un_ext)**2
            limitern = sqrt(g2n_ext[:-1:] + g2n_ext[1::])
            fluxn = (fn_ext[:-1:] + fn_ext[1::])/2. \
                  - limitern * (un_ext[1::] - un_ext[:-1:])/2.
            u0_ext = hstack([u0[0], u0, u0[-1]])
            f0_ext = self.flux.fluxfun(u0_ext)
            g20_ext = self.flux.fluxder(u0_ext)**2
            limiter0 = sqrt(g20_ext[:-1:] + g20_ext[1::])
            flux0 = (f0_ext[:-1:] + f0_ext[1::])/2. \
                  - limiter0 * (u0_ext[1::] - u0_ext[:-1:])/2.
            res = (un - u0)/dt + (fluxn[1::]-fluxn[:-1:])/self.dx/2.\
                + (flux0[1::]-flux0[:-1:])/self.dx/2. - force
        else:
            print 'scheme unidentified'
            exit()            
        return res
        
    def tintegral(self):
        self.J = None
        self.utx = self.uref[0].reshape([1,self.N])
        self.ts = self.tref[0].reshape([1,])

        dt = self.tref[1]-self.tref[0]
        tnow = self.ts[0]
        while tnow._base < self.tfinal:
            adsol = solve(self.residual, self.utx[-1],\
                    args=(self.utx[-1], dt), \
                    kargs={'scheme':'Trapozoidal',\
                           'force':self.force},\
                    max_iter=6, verbose=False)
            if adsol is None:
                return False
            tnow += dt
            self.utx = vstack([self.utx,adsol.reshape([1,adsol.size])])
            self.ts = hstack([self.ts, tnow])
        return True

    def objective(self, u_obj=None):
        '''
        if u_obj is None: compare self.utx with self.uref
        otherwise: compare self.utx with u_obj
        '''
        converged = self.tintegral()
        if not converged:
            self.J = 1e16
        else:
            if u_obj is None:
                self.J = sum((self.utx - self.uref)**2)*.1/100
            else:
                self.J = sum((self.utx - u_obj)**2)*.1/100
        return self.J


if __name__ == '__main__':
     unittest.main()
#    mtwin = multimodels(nsol=1)
#    coef = np.random.random(mtwin.models[0].flux.ns)
#    opt = nlopt.opt(nlopt.LD_SLSQP, coef.size)
#    opt.set_min_objective( mtwin.func_grad )
#    opt.set_stopval(1e-4)
#    opt.set_maxeval(200)
#    xcoef = opt.optimize(coef)
