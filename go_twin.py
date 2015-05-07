"Goal-oriented twin model gradient estimation"
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
        self.xs = np.linspace(0,1,101)
        self.uinit = np.sin(2.*np.pi*self.xs)/2. + .5
        self.uinit[20:40] = 0.
        self.uinit[70:80] = 1.
        self.tinit = 0.
        self.tfinal = .1
        self.source = array([0.3, -0.3])
        self.tgrid = linspace(self.tinit, self.tfinal, 100)
        # primal parameters
        self.A = 2.
        # twin parameters
        self.nsig = 10
        self.rangesig = [0., 1.1]
  
    @unittest.skipIf(True, '')
    def test_primal(self):
        primal = \
        PrimalModel(self.uinit, self.xs, self.tinit, self.tfinal, self.A)
        primal.set_source(self.source)
        primal.integrate(self.tfinal)
        set_trace()
        plt.clf()
        primal.plot_utx(self.tgrid)
        set_trace()

    @unittest.skipIf(True, '')
    def test_twin(self):
        twin = \
        TwinModel(self.uinit, self.xs, self.tinit, self.tfinal, self.nsig, self.rangesig)
        twin.set_source(self.source)
        twin.flux.setcoef(np.loadtxt('xcoef'))
        twin.integrate(self.tfinal)
        plt.clf()
        twin.plot_utx(self.tgrid)

    @unittest.skipIf(True, '')
    def test_mismatch(self):
        coef = np.loadtxt('xcoef')
        infertwin = \
        InferTwinModel(self.xs, self.uinit, self.tinit, self.tfinal, self.source,
                       self.A, self.nsig, self.rangesig, coef)
        lasso_reg = 1e-4
        grad = np.zeros(coef.size)
        val0 = infertwin.var_grad(coef, grad, lasso_reg, infertwin.primal.tfinal)
        infertwin.clean()

        dcoef = zeros(coef.shape)
        dcoef[5] += 1e-4
        infertwin.twin.flux.setcoef(coef+dcoef)
        infertwin.twin.set_source(self.source)
        val1 = infertwin.var_grad(coef+dcoef, grad, lasso_reg, infertwin.primal.tfinal)
        print (val1-val0)/1e-4, grad[5]

    @unittest.skipIf(True, '')
    def test_infer(self):
        coef = np.loadtxt('coef/xcoef_final')
        infertwin = \
        InferTwinModel(self.xs, self.uinit, self.tinit, self.tfinal, self.source,
                       self.A, self.nsig, self.rangesig, coef)
        lasso_reg = 1e-4
        trained_coef = infertwin.infer(coef, lasso_reg)

    @unittest.skipIf(False, '')
    def test_goaloriented(self):
        # infer flux with uniform space-time weighting
        trained_coef = np.loadtxt('coef/xcoef_final') 
        lasso_reg = 1e-4
        # determine active basis
        active_list = np.abs(degrade(trained_coef)) > 1e-5

        ensemble_coef = trained_coef.copy() 
        for i in np.r_[7:8]: #range(trained_coef.size):
            if not active_list[i]:
                continue
            dcoef = np.zeros(trained_coef.size)
            dcoef[i] = .2
            ensemble_coef = np.vstack([degrade(ensemble_coef), degrade(trained_coef+dcoef)])
            ensemble_coef = np.vstack([degrade(ensemble_coef), degrade(trained_coef-dcoef)])

        go_weight = GoalOrientedWeight(self.xs, self.uinit, self.tinit, self.tfinal,
                    self.source, self.A, self.nsig, self.rangesig, active_list, ensemble_coef)

        w_param = array([1.,0.,0.])
        grad = np.zeros(3)
        var0 = go_weight.var_dJdc(w_param, grad)

        w_param = array([1.,0.+1e-3,0.])
        var1 = go_weight.var_dJdc(w_param, grad)
        print var1-var0
        print grad[1]*1e-3
        

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
        us = linspace(degrade(lend), degrade(rend), 1000)
        if not grad:
            y = degrade(self.fluxfun(us))
        else:
            y = degrade(self.fluxder(us))
        handle, = plt.plot(degrade(us), y, color=cl)
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
        self.source_raw = None
        self.source = None
        self.profiles = None
        self.flux = None
        self.utx = uinit[np.newaxis,:]
        self.ts = np.array(tinit)

    def set_source(self, source):
    # set space dependent design (source)
    # source is constant in time, modelled by bubble profiles in space
        if isinstance(source, np.ndarray):
            source = upgrade(source)
        dim = source.size
        location = np.linspace(0,1,dim)
        distance = location[1] - location[0]
        profiles = \
        [exp( -(self.xs-center)**2/ distance**2 ) for center in location]
        self.profiles = profiles
        self.source_raw = source
        self.source = sum( [profiles[ii] * self.source_raw[ii] for ii in range(dim)], 0 )

    def residual(self, un, u0, dt):
        # one timestep residual
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
        dt = (np.min([self.tfinal, tcutoff]) - self.tinit)/50
        mindt = dt/2e2
        endt = np.min([self.tfinal, tcutoff])
        print '-'*40
        while tnow<endt:
            print tnow
            adsol = solve(self.residual, self.utx[-1], \
                          args = (self.utx[-1], dt), \
                          max_iter=100, verbose=False)
            tnow += dt
            self.utx = vstack([self.utx, adsol.reshape([1,adsol.size])])
            self.ts = hstack([self.ts, np.array(tnow)])
            if adsol._n_Newton < 4:
                dt *= 2.
            elif adsol._n_Newton < 12:
                pass
            elif adsol._n_Newton < 64 and dt>mindt:
                dt /= 2.
            else:
                return False
        return True

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
        plt.contourf(T,X,degrade(utx_grid), 20)


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
# infer design/source dependent twin model

    def __init__(self, xs, uinit, tinit, tfinal, source, 
                 A, nsig, rangesig, coef=None):
        # solve primal model for reference solution on tgrid
        self.primal = PrimalModel(uinit, xs, tinit, tfinal, A)
        self.primal.set_source(source)
        self.primal.integrate(tfinal)
        # initialize twin model
        self.twin = TwinModel(uinit, xs, tinit, tfinal, nsig, rangesig)
        self.twin.set_source(source)
        if coef is None:
            coef = np.loadtxt('xcoef')
        self.twin.flux.setcoef(coef.copy())
        self.last_working_coef = coef.copy()
        self.u_target = None

    def clean(self):
        self.twin.utx.obliviate()
        self.twin.source.obliviate()
        self.twin.flux.coef.obliviate()
        if self.twin.utx.shape[0]>1:
            self.twin.utx = self.twin.utx[0][np.newaxis,:].copy()

    def mismatch(self, lasso_reg, tcutoff):
        # solution mismatch in [0,tcutoff], with Lasso basis selection
        # map twin model solution to primal model's time grid
        if not self.twin.integrate(tcutoff):
            return False
        tgrid = linspace(self.primal.tinit, np.min([self.primal.tfinal, tcutoff]), 
                         1+np.ceil(50.*tcutoff/self.primal.tfinal))
        uprimal = self.primal.interp_tgrid(tgrid)
        utwin   = self.twin.interp_tgrid(tgrid)
        self.u_target = utwin[-1]
        sol_mismatch = linalg.norm(uprimal-utwin,2)**2
        reg = linalg.norm(self.twin.flux.coef, 1)

        self.last_working_coef = degrade(self.twin.flux.coef).copy()
        return sol_mismatch + lasso_reg * reg

    def var_grad(self, coef, grad, lasso_reg, tcutoff):
        # solution mismatch value and gradient
        self.twin.flux.setcoef(coef.copy())
        val = self.mismatch(lasso_reg, tcutoff)
        if isinstance(val, bool):
            val = 1e10
            grads = .1/(coef-self.last_working_coef)[np.newaxis,:]
        else:
            grads = val.diff(self.twin.flux.coef)
            val.obliviate()
        for i in range(self.twin.flux.coef.size):
            grad[i] = grads[0,i]

        print tcutoff, 'val: ', degrade(val)
        self.clean()
        return float(degrade(val))

    def infer(self, coef, lasso_reg):
        # optimize selected basis coefficients
        for tcutoff in np.logspace(-3,0,5)*self.primal.tfinal:
            opt = nlopt.opt(nlopt.LD_LBFGS, coef.size)
            opt.set_min_objective(lambda coef, grad: 
                                  self.var_grad(coef, grad, lasso_reg, tcutoff))
            opt.set_stopval(1e-1)
            opt.set_ftol_rel(1e-2)
            opt.set_maxeval(100)
            if tcutoff == self.primal.tfinal:
                opt.set_stopval(0.)
                opt.set_ftol_rel(1e-4)
            coef = opt.optimize(degrade(coef).copy())
        return coef


'Goal-oriented weight for twin model inference'

class GoalOrientedWeight:
# infer optimal weighting scheme for twin model inference

    def __init__(self, xs, uinit, tinit, tfinal, source, A, nsig, rangesig, 
                 active_list, ensemble_coef):
        self.xs = xs
        self.uinit = uinit
        self.tinit = tinit
        self.tfinal = tfinal
        self.tgrid = linspace(self.tinit, self.tfinal, 100)
        self.source = source
        self.A = A
        self.nsig = nsig
        self.rangesig = rangesig
        self.active_list = active_list
        self.ensemble_coef = array(ensemble_coef)

        self.primal = PrimalModel(uinit, xs, tinit, tfinal, A)
        self.primal.set_source(source)
        self.primal.integrate(self.tfinal)
        self.uprimal = self.primal.interp_tgrid(self.tgrid)
        
        self.w_param = None     # parameters of weight (len 3)
        self.w_primal = None    # weights on primal's space-time solution

    def update_weights(self, w_param):
        self.w_param = w_param
        w0 = self.w_param[0] * ones(self.uprimal.shape)
        w1 = self.w_param[1] * 2. * abs_smooth(self.uprimal)
        w2 = self.w_param[2] * 12. * (self.uprimal-.5)**2
        w = w0+w1+w2
        self.w_primal = w/sum(w)

    def likelihood(self, utwin):
        # likelihood of space-time solution fitting given coef
        # weight_params: [0]:1 [1]:2*|u| [2]:12*(u-.5)^2
        # the discretized realized weights sum to one
        if self.w_primal is None:
            print 'weight not initiated'
            exit(1)
        return exp( - linalg.norm( (self.uprimal - utwin) * self.w_primal, 2) )


    def var_dJdc(self, w_param, grad):
        # ensemble variance of dJ/dc and its gradient to weight_params
        # objective is to match the flow at t=tfinal to u=.5
        self.update_weights(w_param)
        dJdc = zeros([self.ensemble_coef.shape[0], self.source.size])
        like = zeros(self.ensemble_coef.shape[0])
        twins = []
        utwins = []
        for i in range(self.ensemble_coef.shape[0]):
            twins.append( TwinModel(self.uinit, self.xs, self.tinit, self.tfinal, 
                                    self.nsig, self.rangesig) )
            twins[-1].set_source(self.source)
            twins[-1].flux.activate(self.active_list)
            twins[-1].flux.setcoef(self.ensemble_coef[i])

            twins[-1].integrate(self.tfinal)
            utwins.append( twins[-1].interp_tgrid(self.tgrid) )
            Ji = linalg.norm(utwins[-1][-1]-.5,2)
            dJdc[i] = Ji.diff(twins[-1].source_raw)
            like[i] = self.likelihood(utwins[-1])
        for i in range(self.ensemble_coef.shape[0]):
            like[i] = like[i] / sum(like)
        dJdc_mean = sum(dJdc*like[i])
        var = array(0.)
        for i in range(self.ensemble_coef.shape[0]):
            var += linalg.norm(dJdc[i] - dJdc_mean, 2)**2 * like[i]
        grads = var.diff(self.w_param)
        for i in range(3):
            grad[i] = grads.todense()[0,i]
        return float(degrade(var))

    def optimize_weight(self):
        pass



if __name__ == '__main__':
    unittest.main()
