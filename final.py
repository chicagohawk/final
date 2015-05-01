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
        self.uinit = np.sin(np.pi*self.xs)/2. + .5
        self.tinit = 0.
        self.tfinal = 1.
        self.source = array([0.3, -0.3])
        self.tgrid = linspace(self.tinit, self.tfinal, 100)
        # primal parameters
        self.A = 2.
        # twin parameters
        self.nsig = 10
        self.rangesig = [-.1, 1.1]
  
    @unittest.skipIf(True, '')
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
        coef = np.zeros(self.nsig)
        infertwin = \
        InferTwinModel(self.xs, self.uinit, self.tinit, self.tfinal, self.source,
                       self.A, self.nsig, self.rangesig, coef)
        lasso_reg = 1e-4
        trained_coef = infertwin.infer(coef, lasso_reg)
    
    @unittest.skipIf(True, '')
    def test_Matern(self):
        mat = Matern(1., 1.)
        cs = upgrade(np.random.rand(2,10))
        c0,c1 = cs[0],cs[1]
        K0 = mat.K0(c0, c1)
        dc = np.random.rand(1,10)*1e-5
        Kd0 = mat.K0(c0+dc, c1)
        print np.dot(K0.diff(c0).todense().view(np.ndarray)[0], dc[0])
        print Kd0-K0

        dc = np.random.rand(1,10)*1e-5
        Kd1 = mat.K0(c0, c1+dc)
        print np.dot(mat.K1(c0,c1)._value, dc[0])
        print Kd1-K0

        dc0 = np.random.rand(1,10)*1e-5
        print mat.K1(c0+dc0,c1) - mat.K1(c0,c1)
        print dot(dc0, mat.K2(c0,c1))
        set_trace()

    @unittest.skipIf(True, '')
    def test_mle(self):
        dimc = 2
        bayes = BayesOpt(dimc)
        cx = np.linspace(-1.,1.,3)
        cy = np.linspace(-1.,1.,3)
        CX, CY = np.meshgrid(cx, cy)
        obj = np.sin(CX+1.2*CY).ravel()
        cs = np.array( zip(CX.ravel(), CY.ravel()) )
        grad = np.array( zip(np.cos(CX+1.2*CY).ravel(), 1.2*np.cos(CX+1.2*CY).ravel()) )
        grad += (np.random.rand(grad.size).reshape(grad.shape)-.5)
        
        for i in range(obj.size):
            bayes.add_data( cs[i], np.array([obj[i]]), grad[i] )

        params = np.array([ 1., 1., 1., .1])
        params = bayes.mle(params,maxiter=2000)
        set_trace()

    @unittest.skipIf(True, '')
    def test_posterior_and_acquisition(self):
        dimc = 1
        bayes = BayesOpt(dimc)
        c = np.linspace(-1., 1.1, 6)
        obj = c**2
        grad = 2*c+np.random.randn(c.size)*.1
        for i in range(obj.size):
            bayes.add_data( np.array([c[i]]), np.array([obj[i]]), np.array([grad[i]]))
        params = np.array([1., .2, .5, .02])

        print 'posterior test:'
        nextc0 = array(0.2)
        muc0, sigc0 = bayes.posterior(nextc0, params)

        nextc1= array(0.2+1e-5)
        muc1, sigc1 = bayes.posterior(nextc1, params)
        print muc1-muc0
        print muc0.diff(nextc0)[0,0] * 1e-5
        print sigc1-sigc0
        print sigc0.diff(nextc0)[0,0] * 1e-5

        test_num = 101
        muc = zeros(test_num)
        sigc = zeros(test_num)
        nextc = np.linspace(-1.1, 1.1, test_num)

        for i in range(test_num):
            c = array(nextc[i])
            muc[i], sigc[i] = bayes.posterior(c, params)
            muc[i].obliviate()
            sigc[i].obliviate()
        muc = degrade(muc)
        sigc = degrade(sigc)

        plt.figure()
        plt.plot(nextc, nextc**2, color='black')
        plt.plot(nextc, muc, linestyle='--', color='black')
        plt.fill_between(nextc, muc+sigc, muc-sigc, alpha=.5, edgecolor='#FF9848',
                         facecolor='#FF9848')

        print 'acquisition test:'
        EI = np.zeros(test_num)
        grads = np.zeros(test_num)
        nextc = upgrade(nextc)
        for i in range(test_num):
            c = array(nextc[i])
            gradi = np.zeros(dimc)
            EI[i] = bayes.acquisition(c, gradi, params)
            grads[i] = gradi
        plt.fill_between(nextc, EI*10., np.zeros(test_num), alpha=.5, edgecolor='#0000FF',
                         facecolor='#0011FF')
        plt.plot(nextc, grads, color='blue', linestyle='--')
        plt.show()
        set_trace()

    @unittest.skipIf(True, '')
    def test_next_design(self):

        print 'DIM 1 TEST'
        dimc = 1
        bayes = BayesOpt(dimc)
        c = np.linspace(-1., 1.1, 6)
        obj = c**2
        grad = 2*c+np.random.randn(c.size)*.1
        for i in range(obj.size):
            bayes.add_data( np.array([c[i]]), np.array([obj[i]]), np.array([grad[i]]))
        params = np.array([1., .2, .5, .02])

        nextc, maxEI = bayes.next_design(params)
        print 'next design: ', nextc
        print 'max EI: ', maxEI

        print 'DIM 2 TEST'
        dimc = 2
        bayes = BayesOpt(dimc)
        cx = np.linspace(-1.,1.,4)
        cy = np.linspace(-1.,1.,4)
        CX, CY = np.meshgrid(cx, cy)
        obj = (CX**2+CY**2).ravel()
        cs = np.array( zip(CX.ravel(), CY.ravel()) )
        grad = np.array( zip(2.*CX.ravel(), 2.*CY.ravel()) )
        grad += np.random.rand(grad.size).reshape(grad.shape) * .1
        
        for i in range(obj.size):
            bayes.add_data( cs[i], np.array([obj[i]]), grad[i] )

        params = np.array([ 1., .2, 1., .2])
        nextc, maxEI = bayes.next_design(params)
        print 'next design: ', nextc
        print 'max EI: ', maxEI
        set_trace()

 

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
                opt.set_ftol_rel(1e-5)
            coef = opt.optimize(degrade(coef).copy())
        return coef

'Matern kernel'
class Matern:

    def __init__(self, sig, rho):
        self.sig = upgrade(sig)
        self.rho = upgrade(rho)

    def update_param(self, sig, rho):
        self.sig = upgrade(sig)
        self.rho = upgrade(rho)

    def K0(self, c0, c1):
        # scalar return
        d = linalg.norm(c0-c1,2)
        return \
        self.sig**2 * (1+np.sqrt(5.)*d/self.rho+5./3*d**2/self.rho**2) \
        * exp(-np.sqrt(5)*d/self.rho)

    def K1(self, c0, c1):
        # vector return
        d = linalg.norm(c0-c1,2)
        return \
        self.sig**2 * exp(-np.sqrt(5.)*d/self.rho) \
        * (5./3/self.rho**2 + 5*np.sqrt(5.)/3*d/self.rho**3) \
        * (c0-c1)

    def K2(self, c0, c1):
        # matrix return
        diffc = upgrade(c0-c1)
        d = linalg.norm(diffc,2)
        matrix = dot(diffc[np.newaxis,:].transpose() , diffc[np.newaxis,:])
        return \
        self.sig**2 * exp(-np.sqrt(5.)*d/self.rho) * \
        (  (5./3/self.rho**2 + 5.*np.sqrt(5.)/3*d/self.rho**3) * eye(diffc.size)
           - 25./3/self.rho**4*matrix 
        )


'Bayesian optimization'
class BayesOpt:

    def __init__(self, dimc):
        self.c_list = []        # design list
        self.obj_list = []      # objective function evaluation list
        self.grad_list = []     # estimated gradient evaluation list
        self.best_index = None  # current best design index in list 

        self.dimc = dimc
        self.obj_kernel = None
        self.err_kernel = None
        self.like_matrix = None    # ndarray
        self.mu = None             # ndarray

    def add_data(self, c, obj, grad):
        self.c_list.append(degrade(c))
        self.obj_list.append(degrade(obj))
        self.grad_list.append(degrade(grad))
        self.best_index = np.argmin(np.hstack(self.obj_list))

    def update_kernel(self, sig, sige, rho, rhoe):
        self.obj_kernel = Matern(sig, rho.copy())
        self.err_kernel = Matern(sige, rhoe.copy())

    def likelihood(self, params, grad, verbose=False):
        # construct data likelihood matrix and mean, evaluate data -1*likelihood
        # ndarray output
        sig, sige, rho, rhoe = params[0], params[1], params[2], params[3]
        self.update_kernel(sig, sige, rho, rhoe)

        like_matrix = np.zeros([len(self.obj_list)*(self.dimc+1), 
                                len(self.obj_list)*(self.dimc+1)])
        for i in range(len(self.c_list)):
            ci = self.c_list[i]
            istart = len(self.c_list)+i*self.dimc
            iend   = istart + self.dimc
            for j in range(len(self.c_list)):
                cj = self.c_list[j]
                jstart = len(self.c_list)+j*self.dimc
                jend   = jstart + self.dimc
                # fill K0
                like_matrix[i,j] = degrade( self.obj_kernel.K0(ci, cj) )
                # fill K1
                like_matrix[i,jstart:jend] = degrade( self.obj_kernel.K1(ci,cj) )
                # fill K2 obj
                like_matrix[istart:iend, jstart:jend] = degrade( self.obj_kernel.K2(ci, cj) )
                # fill K2 err
                like_matrix[istart:iend, jstart:jend] += \
                degrade( self.err_kernel.K0(ci, cj) * np.eye(self.dimc) )

        like_matrix = np.triu(like_matrix,1).transpose() + np.triu(like_matrix)

        obj_list = np.hstack(self.obj_list)
        grad_list = np.hstack(self.grad_list)
        datavec = np.hstack([obj_list, grad_list])

        # posterior mean of objective
        matrix = like_matrix[:len(self.obj_list),:len(self.obj_list)]
        data   = datavec[:len(self.obj_list)]
        try:
            mu_obj = np.sum( np.linalg.solve( matrix, data ) ) / \
                     np.sum( np.linalg.solve( matrix, np.ones(len(obj_list)) ) )
        except:
            return 1e5

        # posterior mean of grads
        mu_grad = np.zeros(self.dimc)
        for i in range(self.dimc):
            matrix = like_matrix[len(self.obj_list)+i::self.dimc, len(self.obj_list)+i::self.dimc]
            data   = datavec[len(self.obj_list)+i::self.dimc]
            mu_grad[i] = np.sum( np.linalg.solve( matrix, data ) ) / \
                         np.sum( np.linalg.solve( matrix, np.ones(len(obj_list)) ) )
        mu = np.hstack([mu_obj, mu_grad])
        self.mu = mu

        self.like_matrix = like_matrix
        mu = np.tile(mu,[len(obj_list),1])
        mu = np.ravel(mu.transpose())
        like_det = np.linalg.det(like_matrix)
        neg_like_eval = \
        np.dot(datavec-mu, np.linalg.solve(like_matrix, datavec-mu)) + np.log(like_det)
        
        if verbose:
            print 'LK:   ',neg_like_eval
            print 'param ', params
        return neg_like_eval
    
    def mle(self, params, maxiter=100):
        opt = nlopt.opt(nlopt.LN_COBYLA, params.size)
        opt.set_min_objective(self.likelihood)
        opt.set_maxeval(maxiter)
        opt.set_lower_bounds(np.zeros( params.size) )
        opt.set_initial_step(np.linalg.norm(params))
        opt.set_ftol_rel(1e-3)
        params = opt.optimize( params )
        return params

    def posterior(self, c, params):
        # posterior evaluation, adarray output
        self.likelihood(params, None)
        vec = []
        for i in range(len(self.c_list)):
            ci = self.c_list[i]
            vec.append( self.obj_kernel.K0(c,ci) )
        for i in range(len(self.c_list)):
            ci = self.c_list[i]
            istart = len(self.c_list)+i*self.dimc
            iend = istart+self.dimc
            vec.append( self.obj_kernel.K1(c, ci) )
        vec = hstack(vec)
        mu_data = np.tile(self.mu,[len(self.obj_list),1])
        mu_data = np.ravel(mu_data.transpose())

        datavec = hstack([hstack(self.obj_list), hstack(self.grad_list)])
        muc = self.mu[0] + \
              dot( vec,
              linalg.solve(self.like_matrix, datavec-mu_data)
              )
        sigc = params[0] - dot(vec, linalg.solve(self.like_matrix, vec))
        return muc, sigc

    def acquisition(self, cnd, grad, params, scheme='EI'):
        # evaluate EI acquisition function and its gradient to c
        if self.best_index is None:
            print 'posterior initialization required'
            exit(1)
        c = upgrade(cnd)
        muc, sigc = self.posterior(c, params)

        if scheme=='EI':
            if sigc._value>1e-10:
                zc = (self.obj_list[self.best_index] - muc) / sigc
                EI = sigc * ( zc/2 * (1+erf(zc/np.sqrt(2))) +
                              1./np.sqrt(2*np.pi)*exp(-zc**2/2) )
            else:
                EI = self.obj_list[self.best_index] - muc
        elif scheme=='UCB':
            EI = - muc + 3.*sigc
        else:
            print 'scheme not recognized'
            exit(1)

        print 'EI: ', EI._value
        EI_grad = EI.diff(c)
        EI.obliviate()
        for i in range(self.dimc):
            grad[i] = degrade(EI_grad[0,i])
        
        return float(degrade(EI))

    def next_design(self, params):
        # next candidate design
        opt = nlopt.opt(nlopt.LD_TNEWTON_PRECOND_RESTART, self.dimc)
        opt.set_max_objective( lambda c, grad:
            self.acquisition(c, grad, params) )
        opt.set_stopval(1e5)
        opt.set_maxeval(50)

        agent_num = 10
        agent_best_c = None
        agent_best_val = -1.
        for i in range(agent_num):
            print 'agent', i
            agent_init = np.array(self.c_list[self.best_index]) \
                       + np.random.randn(self.dimc)*.1
            nextc = opt.optimize(agent_init.copy())
            if opt.last_optimum_value() > agent_best_val:
                agent_best_c = nextc
                agent_best_val = opt.last_optimum_value()

        return agent_best_c, agent_best_val


def optimize_control():
    pass


if __name__ == '__main__':
    unittest.main()
