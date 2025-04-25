import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from .me_forward import MEForward
from .annealing import DualAnnealing
class MEInversion(MEForward):
    # Default range of parameters, ref: 10.1007/s11207-014-0497-7 (Centeno et al. 2014)
    v_D_range = [1.,500.]
    v_los_range = [-7.e3,7.e3]
    eta_0_range = [1.,1000.]
    S10_range = [0.1,10.]
    a_damp_range = [0.4,0.6]
    Bmag_range = [5.,5000.]
    theta_range = [0.,np.pi]
    phi_range = [0.,np.pi]
    def __init__(self,wavebands,landeG=2.5,lambda0=630.25,wing=None):
        super().__init__(wavebands,landeG,lambda0,wing=wing)
        self.evaluator = None
        self.optimizer = None
        self.ivs_method = None
        self.ivs_results = None
        self.obsevation = dict()

    def synthesize(self,params_tensor:torch.Tensor):
        v_D,v_los,eta_0,S10,a_damp,Bmag,theta,phi = params_tensor.unsqueeze(0).permute(2,1,0)
        I,Q,U,V = self.return_IQUV(v_D,v_los,eta_0,S10,a_damp,Bmag,theta,phi)
        return torch.stack([I,Q,U,V],dim=1)

    def merit_function(self,iquv_obs:torch.Tensor,iquv_syn:torch.Tensor):
        wights = torch.tensor([1,5,5,3.5],device=iquv_obs[0].device,dtype=iquv_obs[0].dtype)
        sigmas = torch.tensor([0.118]+[0.204]*3,device=iquv_obs[0].device,dtype=iquv_obs[0].dtype)
        F = iquv_obs.size(1)*iquv_obs.size(2)-8
        chi2   = torch.sum((iquv_obs-iquv_syn)**2/sigmas[None,:,None]**2*wights[None,:,None]**2,dim=(-1,-2)).unsqueeze(1)
        return chi2/F

    def gradient(self, y, x):
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=False,
            retain_graph=False
        )[0]
        return gradients

    def adam_gradient(self,t,f,x,mo,vo,**kwargs):
        beta1 = kwargs.get('beta1',0.9)
        beta2 = kwargs.get('beta2',0.999)
        epsilon = kwargs.get('epsilon',1e-8)
        gt = self.gradient(f,x)
        mt = beta1*mo+(1-beta1)*gt
        vt = beta2*vo+(1-beta2)*gt**2
        mt_hat = mt/(1-beta1**t)
        vt_hat = vt/(1-beta2**t)
        ret = mt_hat/(vt_hat**0.5+epsilon)
        return ret,mt,vt

    def normalizing_parameter(self,params_tensor:torch.Tensor):
        v_D_range = self.v_D_range
        v_los_range = self.v_los_range
        eta_0_range = self.eta_0_range
        S10_range = self.S10_range
        a_damp_range = self.a_damp_range
        Bmag_range = self.Bmag_range
        theta_range = self.theta_range
        phi_range = self.phi_range
        # sigma = (v_los_range[1]-v_los_range[0])/6
        v_D,v_los,eta_0,S10,a_damp,Bmag,theta,phi = params_tensor.unsqueeze(0).permute(2,1,0)
        xn1 = (torch.log(v_D*1e4)-np.log(v_D_range[0]))/(np.log(v_D_range[1])-np.log(v_D_range[0]))
        xn2 = (v_los-v_los_range[0])/(v_los_range[1]-v_los_range[0])
        xn3 = (torch.log(eta_0)-np.log(eta_0_range[0]))/(np.log(eta_0_range[1])-np.log(eta_0_range[0]))
        xn4 = (torch.log(S10)-np.log(S10_range[0]))/(np.log(S10_range[1])-np.log(S10_range[0]))
        xn5 = (torch.log(a_damp)-np.log(a_damp_range[0]))/(np.log(a_damp_range[1])-np.log(a_damp_range[0]))
        xn6 = (torch.log(Bmag)-np.log(Bmag_range[0]))/(np.log(Bmag_range[1])-np.log(Bmag_range[0]))
        xn7 = theta/np.pi
        xn8 = phi/np.pi
        return torch.cat([xn1,xn2,xn3,xn4,xn5,xn6,xn7,xn8],dim=1)
    
    def denormalizing_parameter(self,xn:torch.Tensor):
        v_D_range = self.v_D_range
        v_los_range = self.v_los_range
        eta_0_range = self.eta_0_range
        S10_range = self.S10_range
        a_damp_range = self.a_damp_range
        Bmag_range = self.Bmag_range
        theta_range = self.theta_range
        phi_range = self.phi_range
        sigma = (v_los_range[1]-v_los_range[0])/6
        xn1,xn2,xn3,xn4,xn5,xn6,xn7,xn8 = xn.split(1,dim=1)
        v_D = torch.exp(xn1*(np.log(v_D_range[1])-np.log(v_D_range[0]))+np.log(v_D_range[0]))/1e4
        v_los = xn2*(v_los_range[1]-v_los_range[0])+v_los_range[0]
        eta_0 = torch.exp(xn3*(np.log(eta_0_range[1])-np.log(eta_0_range[0]))+np.log(eta_0_range[0]))
        S10 = torch.exp(xn4*(np.log(S10_range[1])-np.log(S10_range[0]))+np.log(S10_range[0]))
        a_damp = torch.exp(xn5*(np.log(a_damp_range[1])-np.log(a_damp_range[0]))+np.log(a_damp_range[0]))
        Bmag = torch.exp(xn6*(np.log(Bmag_range[1])-np.log(Bmag_range[0]))+np.log(Bmag_range[0]))
        theta = xn7*torch.pi % np.pi
        phi = xn8*torch.pi % np.pi
        return torch.cat([v_D,v_los,eta_0,S10,a_damp,Bmag,theta,phi],dim=1)

    def plot_annealing_hist(self,**kwargs):
        if self.ivs_method == 'csa':
            T0 = self.optimizer.results['settings']['T0']
            Tf = self.optimizer.results['Tf']
            T_list = np.exp(np.linspace(np.log(T0),np.log(Tf),100))
            Emin_list = np.array(self.optimizer.results['Emin_list'])
            plt.plot(T_list,Emin_list)
            plt.xlabel('Temperature [K]')
            plt.ylabel('Energy')
            plt.yscale('log')
            plt.xscale('log')
            plt.gca().invert_xaxis()
            plt.title('Annealing History')

        elif self.ivs_method == 'gsa':
            T_list = self.results.history['T_list']
            Emin_list = self.results.history['emin_history']
            plt.plot(T_list,Emin_list)
            plt.xlabel('Temperature [K]')
            plt.ylabel('Energy')
            plt.yscale('log')
            plt.xscale('log')
            plt.gca().invert_xaxis()
            plt.title('Annealing History')
        else:
            raise RuntimeError(f"Could not plot before inversion done")
        
    def plot_inversion_results(self, **kwargs):
        """
        Plot the inversion results

        Parameters:
        ===========
        choice_index: int, choice index, default: 0
        params_obs: torch.Tensor, observation parameters, default: None
        """
        choice = kwargs.get('choice_index',0)
        lambda0 = self.lambda0
        lm = self.wavebands[1:].detach().cpu().numpy()
        wing = self.wing
        half_bandwidth = np.abs(lambda0-wing)
        ll = np.linspace(lambda0-half_bandwidth,lambda0+half_bandwidth,1000)
        ll_tensor = torch.from_numpy(ll).to(self.obsevation['iquv_obs'].device).to(self.obsevation['iquv_obs'].dtype)
        forward_cont = MEForward(ll_tensor,lambda0=630.25,wing=self.wing)
        params_obs = kwargs.get('params_obs',None)
        iquv_obs = self.obsevation['iquv_obs']
        if params_obs is not None:
            iquv_con = torch.stack(list(forward_cont(*params_obs.T[:,:,None])),dim=1)
        if self.ivs_method == 'csa':
            params_ivs = torch.from_numpy(self.ivs_results['x']).to(self.obsevation['iquv_obs'].device).to(self.obsevation['iquv_obs'].dtype)
            iquv_ivs = torch.stack(list(forward_cont(*params_ivs.T[:,:,None])),dim=1)
        elif self.ivs_method == 'gsa':
            params_ivs = self.denormalizing_parameter(self.results.x)
            iquv_ivs = torch.stack(list(forward_cont(*params_ivs.T[:,:,None])),dim=1)
        else:
            raise ValueError(f'Could not plot before inversion done')
        plt.subplot(2,2,1)
        if params_obs is not None:
            plt.plot(ll,iquv_con[choice,0].cpu().numpy().squeeze(),label='I_continuous',ls='--')
        plt.plot(ll,iquv_ivs[choice,0].cpu().numpy().squeeze(),label='I_inversion',ls=':')
        plt.plot(lm,iquv_obs[choice,0].cpu().numpy().squeeze(),label='I_observation',ls='',marker='o',ms=5)
        plt.legend()

        plt.subplot(2,2,2)
        if params_obs is not None:
            plt.plot(ll,iquv_con[choice,1].cpu().numpy().squeeze(),label='Q_continuous',ls='--')
        plt.plot(ll,iquv_ivs[choice,1].cpu().numpy().squeeze(),label='Q_inversion',ls=':')
        plt.plot(lm,iquv_obs[choice,1].cpu().numpy().squeeze(),label='Q_observation',ls='',marker='o',ms=5)
        plt.legend()

        plt.subplot(2,2,3)
        if params_obs is not None:
            plt.plot(ll,iquv_con[choice,2].cpu().numpy().squeeze(),label='U_continuous',ls='--')
        plt.plot(ll,iquv_ivs[choice,2].cpu().numpy().squeeze(),label='U_inversion',ls=':')
        plt.plot(lm,iquv_obs[choice,2].cpu().numpy().squeeze(),label='U_observation',ls='',marker='o',ms=5)
        plt.legend()

        plt.subplot(2,2,4)
        if params_obs is not None:
            plt.plot(ll,iquv_con[choice,3].cpu().numpy().squeeze(),label='V_continuous',ls='--')
        plt.plot(ll,iquv_ivs[choice,3].cpu().numpy().squeeze(),label='V_inversion',ls=':')
        plt.plot(lm,iquv_obs[choice,3].cpu().numpy().squeeze(),label='V_observation',ls='',marker='o',ms=5)
        plt.legend()
    
    def __call__(self,iquv_obs:torch.Tensor,**kwargs):
        '''
        Input:
            iquv_obs: torch.Tensor, observed IQUV with shape (B,4,N)
        ====
        Parameters:
            method: str, optimization method, default: 'gsa', ['gsa','csa']
            device: torch.device, device, default: iquv_obs.device
            Other parameters see `CudaAnnealing` if 'csa' or `DualAnnealing` if 'gsa'
        ====
        Output:
            params_ivs: torch.Tensor, inverted parameters
        '''
        self.obsevation['iquv_obs'] = iquv_obs
        method = kwargs.pop('method', 'gsa')
        device = kwargs.get('device',iquv_obs.device)
        self.ivs_method = method
        if method == 'csa':
            kwargs.pop('method', None)
            iquv_obs = iquv_obs.to(device)
            func = lambda x: self.merit_function(iquv_obs,self.synthesize(self.denormalizing_parameter(x)))
            x_guess = kwargs.pop('x_guess', torch.rand(iquv_obs.size(0), 8).float().to(device))
            optimizer = CudaAnnealing(func,x_guess,**kwargs)
            x_best,E_best = optimizer.optimizing(x_guess,**kwargs)
            self.optimizer = optimizer
            self.evaluator = E_best/optimizer.results['E0']
            params_ivs = self.denormalizing_parameter(x_best)
            self.ivs_results = dict(
                x0=self.denormalizing_parameter(x_guess).detach().cpu().numpy(),
                e0=optimizer.results['E0'],
                x=params_ivs.detach().cpu().numpy(),
                e=E_best.detach().cpu().numpy(),
            )
            return params_ivs
        elif method == 'gsa':
            func = lambda x: self.merit_function(iquv_obs,self.synthesize(self.denormalizing_parameter(x)))
            x_guess = kwargs.pop('x_guess', torch.rand(iquv_obs.size(0), 8).float().to(device))
            bounds = [[0.,1.]]*8
            maxiter = kwargs.pop('max_iter', kwargs.pop('maxiter', 1000))
            adam = kwargs.pop('adam', dict())
            initial_temp = kwargs.pop('initial_temp', kwargs.pop('initial_temperature', 5230.))
            visit = kwargs.pop('visit',2.62)
            accept = kwargs.pop('accept',-5.0)
            no_local_search = kwargs.pop('no_local_search',False)
            res = DualAnnealing(func,bounds,x0=x_guess,adam=adam,maxiter=maxiter,initial_temp=initial_temp,
                                visit=visit,accept=accept,no_local_search=no_local_search,**kwargs)
            self.results = res
            params_ivs = self.denormalizing_parameter(res.x)
            self.ivs_results = dict(
                x0=self.denormalizing_parameter(x_guess).detach().cpu().numpy(),
                e0=res.e0.detach().cpu().numpy(),
                x=params_ivs.detach().cpu().numpy(),
                e=res.e.detach().cpu().numpy(),
            )
            return params_ivs
        else:
            raise ValueError(f'Invalid method: {method}, support: csa, gsa')

class CudaAnnealing:
    def __init__(self,func,x0:torch.Tensor,**kwargs):
        self.x0 = x0
        self.results = []
        self.func = func
        self.device = x0.device
        self.results = dict(E0=func(x0),x0=x0)
        self.fixer = None

    def _initialize_optimizer(self,**kwargs):
        T0 = kwargs.get('initial_temperature', 5000)
        kB = kwargs.get('accept', None)
        AR = kwargs.get('accept_ratio', 1e2)
        settings = dict(T0=T0,kB=kB,AR=AR)
        if kB is None:
            settings['kB'] = self.results['E0'].detach().cpu().numpy()/T0*AR
        else:
            settings['accept_ratio'] = kB/self.results['E0'].detach().cpu().numpy()*T0
        settings['ds_min'] = kwargs.get('ds_min', 1e-8)
        settings['ds_i']   = kwargs.get('initial_stepsize', 1e-1)
        settings['ds_f']   = kwargs.get('final_stepsize', 1e-6)
        settings['patience'] = kwargs.get('patience', 10)
        settings['cooling_rate'] = kwargs.get('cooling_rate', 0.90)
        settings['max_iter']   = kwargs.get('max_iter', 1000)
        settings['preference_rate'] = kwargs.get('preference_rate', 0.5)
        settings['ending_temperature'] = kwargs.get('ending_temperature', 1e-4)
        settings['gradient_type'] = kwargs.get('gradient_type', 'automatic')
        self.results['settings'] = settings

    def _is_acceptable(self, dE, T):
        kB = torch.from_numpy(self.results['settings']['kB']).to(self.device).to(dE.dtype)
        accept = torch.zeros_like(dE, dtype=torch.bool, device=self.device)
        # self.fixer = dE
        # accept = torch.where(dE<=0, True, torch.bernoulli(torch.exp(-dE/T/kB)).to(torch.bool))
        accept[dE<=0] = True
        accept[dE> 0] = torch.bernoulli(torch.exp(-dE[dE>0]/T/kB[dE>0])).to(torch.bool)
        return accept
    
    def gradient(self, y, x):
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=False,
            retain_graph=False
        )[0]
        return gradients
    
    def numerical_gradient(self,x):
        eps = 1e-6
        nparams = x.size(1)
        grad = torch.zeros_like(x)
        for i in range(nparams):
            x_plus = x.clone()
            x_plus[:,i] += eps
            x_minus = x.clone()
            x_minus[:,i] -= eps
            grad[:,i] = (self.func(x_plus)-self.func(x_minus))/(2*eps)
        return grad

    def error(self,x):
        x = x.requires_grad_(True)
        y = self.func(x)
        g = self.gradient(y,x)
        err = y/g
        return torch.abs(err)
    
    def _next_step(self,x,ds,**kwargs):
        step_type = kwargs.get('step_type', 'gradient')
        ds_min = self.results['settings']['ds_min']
        gradient_type = kwargs.get('gradient_type', 'automatic')
        ds = np.random.rand()*(np.log(ds)-np.log(ds_min))+np.log(ds_min)
        ds = np.exp(ds)
        if step_type == 'gradient':
            x      = x.requires_grad_(True)
            y      = self.func(x)
            if gradient_type == 'automatic':
                g      = self.gradient(y,x).unsqueeze(-1)
            else:
                g      = self.numerical_gradient(x)
            g      = g/torch.norm(g,dim=1,keepdim=True)
            e      = kwargs.get('preference_rate', 0.5)
            rdir   = 2*torch.randn(*x.shape,100,device=x.device,dtype=x.dtype)-1
            rdir   = rdir/torch.norm(rdir,dim=1,keepdim=True)
            judge  = (1-e)/(1-e*torch.sum(rdir*g,dim=1))
            luck   = torch.rand(x.shape[0],100,device=x.device,dtype=x.dtype)
            mask   = judge>luck
            first  = torch.argmax(mask.int(),dim=-1)
            first  = first[:,None,None].repeat(1,8,1)
            direction = -torch.gather(rdir,2,first).squeeze()
        elif step_type == 'random':
            direction = 2*torch.randn(*x.shape,device=x.device,dtype=x.dtype)-1
            direction = direction/torch.norm(direction,dim=1,keepdim=True)
        elif step_type == 'adam':
            x = x.requires_grad_(True)
            if gradient_type == 'automatic':
                g = self.gradient(self.func(x),x)
            else:
                g = self.numerical_gradient(x)
            adam_config = kwargs.get('adam_config', None)
            if adam_config is None:
                raise ValueError('adam_config is required, keys:mo,vo,t | beta1,beta2,epsilon')
            mo = adam_config.get('mo', torch.zeros_like(x))
            vo = adam_config.get('vo', torch.zeros_like(x))
            t  = adam_config.get('t', 0)
            beta1 = adam_config.get('beta1', 0.9)
            beta2 = adam_config.get('beta2', 0.999)
            epsilon = adam_config.get('epsilon', 1e-8)
            ga,mo,vo = self.adam_gradient(t,g,mo,vo,beta1=beta1,beta2=beta2,epsilon=epsilon)
            ga = -ga.detach().unsqueeze(-1)
            ga = ga/torch.norm(ga,dim=1,keepdim=True)
            e      = kwargs.get('preference_rate', 0.5)
            rdir   = 2*torch.randn(*x.shape,100,device=x.device,dtype=x.dtype)-1
            rdir   = rdir/torch.norm(rdir,dim=1,keepdim=True)
            judge  = (1-e)/(1-e*torch.sum(rdir*ga,dim=1))
            luck   = torch.rand(x.shape[0],100,device=x.device,dtype=x.dtype)
            mask   = judge>luck
            first  = torch.argmax(mask.int(),dim=-1)
            first  = first[:,None,None].repeat(1,8,1)
            direction = torch.gather(rdir,2,first).squeeze()
            x.requires_grad_(False)
            step = -direction*ds
            return step,mo,vo
        else:
            raise ValueError('step_type is required, support:gradient,random,adam')
        step   = direction*ds
        x.requires_grad_(False)
        return step
    
    def adam_gradient(self,t,gt,mo,vo,**kwargs):
        beta1 = kwargs.get('beta1',0.9)
        beta2 = kwargs.get('beta2',0.999)
        epsilon = kwargs.get('epsilon',1e-8)
        mt = beta1*mo+(1-beta1)*gt
        vt = beta2*vo+(1-beta2)*gt**2
        mt_hat = mt/(1-beta1**t)
        vt_hat = vt/(1-beta2**t)
        ret = mt_hat/(vt_hat**0.5+epsilon)
        return ret,mt,vt
        
    def _annealing(self,x_guess:torch.Tensor,**kwargs):
        '''
        Input:
            x_guess: torch.Tensor, initial guess of parameters
        ====
        Parameters:
            initial_temperature: float, initial temperature, default: 5000
            accept: float, accept ratio, default: None
            accept_ratio: float, accept ratio, default: 100
            ds_min: float, minimum stepsize, default: 1e-8
            initial_stepsize: float, initial stepsize, default: 1e0
            final_stepsize: float, final stepsize, default: 1e-4
            patience: int, patience, default: 10
            cooling_rate: float, cooling rate, default: 0.90
            max_iter: int, maximum iterations, default: 1000
            preference_rate: float, preference rate, default: 0.5
        ====
        Output:
            x_best: torch.Tensor, best parameters
            E_best: float, best energy
        '''
        start_time = time.time()
        self._initialize_optimizer(**kwargs)
        settings = self.results['settings']
        T0 = settings['T0']
        Tf = settings['ending_temperature']
        CF = settings['cooling_rate']
        nepoch = np.ceil(np.log(Tf/T0)/np.log(CF)).astype(int)
        x_init = x_guess.clone()
        x_best = x_init.clone()
        E_best = self.func(x_init)
        # E_init = E_best.clone()
        x_current = x_init.clone()
        E_current = E_best.clone()
        wait_cnts = torch.zeros(x_current.shape[0],device=self.device,dtype=torch.int32)
        stop_flag = torch.zeros(x_current.shape[0],device=self.device,dtype=torch.bool)
        epoch = 1
        T = T0
        print('### ================================================================== ###')
        print('###                           Begin Annealing                          ###')
        print('### ================================================================== ###')
        time_used = (time.time()-start_time)/60
        step_type = kwargs.get('step_type', 'gradient')
        self.fixer = {}
        print(f' Epoch: {1:05d}, Temperature: {T:9.4f}, Energy: {E_current.min():.6e}, Time: {time_used:7.3f}m')
        self.results['Emin_list'] = [E_init.min().item()]
        while True:
            if T<Tf:
                print(f'Encounter ending temperature: {T:7.2f}')
                break
            if torch.all(stop_flag):
                print(f'Encounter stop flag')
                break
            ds = max(np.sqrt(T/T0)*settings['ds_i'],settings['ds_f'])
            E_init = E_best.clone()
            for iters in range(settings['max_iter']):
                # if torch.all(stop_flag):
                #     break
                if step_type == 'adam':
                    if T==T0 and iters==0:
                        beta1 = kwargs.pop('beta1', 0.9)
                        beta2 = kwargs.pop('beta2', 0.999)
                        epsilon = kwargs.pop('epsilon', 1e-8)
                        mo = torch.zeros_like(x_current)
                        vo = torch.zeros_like(x_current)
                        tt  = 1
                        adam_config = dict(mo=mo,vo=vo,t=tt,beta1=beta1,beta2=beta2,epsilon=epsilon)
                        kwargs['adam_config'] = adam_config
                    dx,mo,vo = self._next_step(x_current,ds,**kwargs)
                    tt += 1
                    kwargs['adam_config']['mo'] = mo
                    kwargs['adam_config']['vo'] = vo
                    kwargs['adam_config']['t']  = tt
                else:
                    dx = self._next_step(x_current,ds,**kwargs)
                x_new = x_current + dx
                if torch.any(x_new>1) or torch.any(x_new<0):
                    x_new = x_new % 1.
                E_new = self.func(x_new)
                # if torch.isnan(E_new).any():
                #     raise ValueError('`E_new` have nan')
                self.results['E_new'] = E_new.detach().cpu().numpy()
                self.results['x_new'] = x_new.detach().cpu().numpy()
                if torch.isnan(E_new).any():
                    # stop_flag[torch.isnan(E_new)[:,0]] = True
                    # print('Calculation encounter nan')
                    E_new[torch.isnan(E_new)[:,0]] = E_current[torch.isnan(E_new)[:,0]]
                    x_new[torch.isnan(E_new)[:,0]] = x_current[torch.isnan(E_new)[:,0]]
                dE = E_new-E_current
                x_best = torch.where(E_new<E_best,x_new,x_best)
                E_best = torch.where(E_new<E_best,E_new,E_best)
                self.results['E_best'] = E_best.detach().cpu().numpy()
                self.results['x_best'] = x_best.detach().cpu().numpy()
                accept = self._is_acceptable(dE,T)[:,0].detach()
                x_current[accept] = x_new[accept]
                E_current[accept] = E_new[accept]
            is_wait = (E_best-E_init)[:,0]>=0
            wait_cnts[~is_wait] = 0
            wait_cnts[is_wait] += 1
            self.fixer['wait_cnts'] = wait_cnts
            stop_flag[wait_cnts> settings['patience']] = True
            stop_flag[wait_cnts<=settings['patience']] = False
            self.fixer['stop_flag'] = stop_flag
            T *= CF
            epoch += 1
            time_used = (time.time()-start_time)/60
            self.results['Emin_list'].append(E_best.min().item())
            print(f' Epoch: {epoch:05d}, Temperature: {T:9.4f}, Energy: {E_best.min():.6e}, Time: {time_used:7.3f}m, Wait: {wait_cnts.min():3d}')
        x_ret = torch.from_numpy(self.results['x_best']).to(self.device)
        E_ret = torch.from_numpy(self.results['E_best']).to(self.device)
        self.results['Tf'] = T
        return x_ret,E_ret

    def _adam(self,x_guess:torch.Tensor,**kwargs):
        start_time = time.time()
        adam_config = kwargs.get('adam',dict())
        nepoch  = adam_config.get('nepoch',1000)
        n_print = adam_config.get('n_print',10)
        lr      = adam_config.get('learning_rate',1e-3)
        if nepoch <=0:
            return x_guess,self.func(x_guess)
        x_init = x_guess.clone()
        x_best = x_init.clone()
        E_best = self.func(x_init)
        E_init = E_best.clone()
        x_current = x_init.clone()
        E_current = E_init.clone()
        print('### ================================================================== ###')
        print('###                              Begin Adam                            ###')
        print('### ================================================================== ###')
        print(f'# Adam #  Rounds: {0:06}/{nepoch:06}, Energy: {E_best.min():.6e}, Wall_time:{(time.time()-start_time)/60:6.2f} [min]')
        beta1 = kwargs.pop('beta1', 0.9)
        beta2 = kwargs.pop('beta2', 0.999)
        epsilon = kwargs.pop('epsilon', 1e-8)
        mo = torch.zeros_like(x_current)
        vo = torch.zeros_like(x_current)
        tt  = 1
        adam_config = dict(mo=mo,vo=vo,t=tt,beta1=beta1,beta2=beta2,epsilon=epsilon)
        kwargs['adam_config'] = adam_config
        for epoch in range(nepoch):
            x_current.requires_grad_(True)
            y_current = self.func(x_current)
            g = self.gradient(y_current,x_current)
            dx,mo,vo = self.adam_gradient(tt,g,mo,vo,**kwargs)
            tt += 1
            kwargs['adam_config']['mo'] = mo
            kwargs['adam_config']['vo'] = vo
            kwargs['adam_config']['t']  = tt
            x_current = x_current - dx*lr
            E_current = self.func(x_current)
            x_best = torch.where(E_current<E_best,x_current,x_best)
            E_best = torch.where(E_current<E_best,E_current,E_best)
            self.results['E_best'] = E_best.detach().cpu().numpy()
            self.results['x_best'] = x_best.detach().cpu().numpy()
            if (epoch+1)%n_print==0 or epoch==nepoch-1:
                time_used = (time.time()-start_time)/60
                print(f'# Adam #  Rounds: {epoch+1:06}/{nepoch:06}, Energy: {E_best.min():.6e}, Wall_time:{time_used:6.2f} [min]')
        x_ret = torch.from_numpy(self.results['x_best']).to(self.device)
        E_ret = torch.from_numpy(self.results['E_best']).to(self.device)
        return x_ret,E_ret
    
    def optimizing(self,x_guess:torch.Tensor,**kwargs):
        x_best,E_best = self._annealing(x_guess,**kwargs)
        x_best,E_best = self._adam(x_best,**kwargs)
        return x_best,E_best
