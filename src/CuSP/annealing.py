import torch
import numpy as np
import time
from scipy.special import gammaln

class VisitDistribution:
    """
    Visit distribution for the annealing process

    Parameters:
    ----------
    lb : tensor_like
        1D tensor, lower bound of the parameter space
    ub : tensor_like
        1D tensor, upper bound of the parameter space
    visiting_param : float
        Parameter for the visiting distribution. Default value is 2.62.
        Higher values give the visiting distribution a heavier tail, this makes the algorithm jump to a more distant region.
        The value range is (1,3]. Its value is fixed for the annealing process.
    """
    TAIL_LIMIT = 1.e8
    MIN_VISIT_BOUND = 1.e-10

    def __init__(self, lb, ub, visiting_param=2.62):
        self._pv = visiting_param
        self.lower = lb.unsqueeze(0) if lb.ndim==1 else lb
        self.upper = ub.unsqueeze(0) if ub.ndim==1 else ub
        self.bound_range = (ub-lb).unsqueeze(0) if ub.ndim==1 else ub-lb
        self.device = None

        # some invariant factors unless visiting_param changes
        self._factor2 = np.exp((4.0-self._pv)*np.log(self._pv-1.0))
        self._factor3 = np.exp((2.0-self._pv)*np.log(2.0)/(self._pv-1.0))
        self._factor4_p = np.sqrt(np.pi)*self._factor2/(self._factor3*(3.0-self._pv))
        self._factor5 = 1.0/(self._pv-1.0)-0.5
        self._d1 = 2.0-self._factor5
        self._factor6 = np.pi*(1.0-self._factor5)/np.sin(np.pi*(1.0-self._factor5))/np.exp(gammaln(self._d1))

    def visiting(self,x,step,temperature):
        """
        Generating new coordinates during the annealing process.
        The new values are based on the visit_fn function.
        when step<dim, changing all coordinates
        when step>=dim, changing one coordinates
        """
        self.device = x.device
        n_batch = x.size(0)
        dim = x.size(1)
        if step<dim:
            # Changing all coordinates
            visits = self.visit_fn(temperature, dim, n_batch)
            upper_sample, lower_sample = torch.rand(2,x.size(0),1).to(x.device).double()
            visits = torch.where(visits>self.TAIL_LIMIT,self.TAIL_LIMIT*upper_sample,visits)
            visits = torch.where(visits<-self.TAIL_LIMIT,-self.TAIL_LIMIT*lower_sample,visits)
            x_visit = x+visits
            a = x_visit-self.lower.to(x.device)
            b = a % self.bound_range.to(x.device)+self.bound_range.to(x.device)
            x_visit = b % self.bound_range.to(x.device)+self.lower.to(x.device)
            x_visit[torch.abs(x_visit-self.lower.to(x.device))<self.MIN_VISIT_BOUND] += 1.e-10
        else:
            # Changing one coordinate
            x_visit = x.clone()
            visit = self.visit_fn(temperature, 1, n_batch)[:,0]
            visit = torch.where(visit>self.TAIL_LIMIT,self.TAIL_LIMIT*torch.rand(x.size(0)).to(x.device).double(),visit)
            visit = torch.where(visit<-self.TAIL_LIMIT,-self.TAIL_LIMIT*torch.rand(x.size(0)).to(x.device).double(),visit)
            index = step-dim
            x_visit[:,index] = visit+x[:,index]
            a = x_visit[:,index]-self.lower[:,index].to(x.device)
            b = a % self.bound_range[:,index].to(x.device)+self.bound_range[:,index].to(x.device)
            x_visit[:,index] = b % self.bound_range[:,index].to(x.device)+self.lower[:,index].to(x.device)
            x_visit[:,index] = torch.where(torch.abs(x_visit[:,index]-self.lower[:,index].to(x.device))<self.MIN_VISIT_BOUND,self.MIN_VISIT_BOUND,x_visit[:,index])
        return x_visit.float()

    def visit_fn(self,temperature,dim,n_batch):
        """
        Ref. https://doi.org/10.1016/S0378-4371(96)00271-3
        """
        x,y = torch.randn(2,n_batch,dim).to(self.device)
        # print(x.shape)
        factor1 = np.exp(np.log(temperature)/(self._pv-1.0))
        factor4 = self._factor4_p*factor1

        x *= np.exp(-(self._pv-1.0)*np.log(self._factor6/factor4)/(3.0-self._pv))
        den = torch.exp((self._pv-1.0)*torch.log(torch.abs(y))/(3.0-self._pv))
        # print(x.shape,den.shape)
        return x/den
        
        
class EnergyState:
    """
    Recording the energy state. Storing the current and the best energy and the corresponding location.

    Parameters:
    ----------
    lower : tensor_like
        1D tensor, lower bound of the parameter space
    upper : tensor_like
        1D tensor, upper bound of the parameter space
    """
    MAX_REINIT_COUNTER = 100
    def __init__(self,lower:torch.Tensor,upper:torch.Tensor):
        self.lower = lower.unsqueeze(0)
        self.upper = upper.unsqueeze(0)
        self.e_current = None
        self.x_current = None
        self.e_best = None
        self.x_best = None
        
    def reset(self, func, **kwargs):
        """
        Reset the energy state. If `x0` is not provided, a random location whith the bounds is generated.
        """
        x0 = kwargs.pop('x0',None)
        if x0 is None:
            n_batch = kwargs.get('n_batch',None)
            if n_batch is None:
                raise ValueError('Either x0 or n_batch must be provided.')
            self.x_current = torch.rand(n_batch,self.lower.size(1)).to(device)*(self.upper-self.lower)+self.lower
            device = self.x_current.device
        elif not isinstance(x0,torch.Tensor):
            raise ValueError('x0 must be a tensor.')
        else:
            self.x_current = x0
            device = x0.device
        init_error = True
        reinit_counter = 0
        while init_error:
            self.e_current = func(self.x_current,**kwargs)
            if self.e_current is None:
                raise ValueError('The energy function must return None')
            if not torch.isfinite(self.e_current).all():
                if reinit_counter > self.MAX_REINIT_COUNTER:
                    raise ValueError('The energy function returned an invalid value. NaN or (+/-) inf')
                self.x_current = torch.rand(n_batch,self.lower.size(1)).to(device)*(self.upper-self.lower)+self.lower
                reinit_counter += 1
            else:
                init_error = False
            # initialize the best energy and location at first step
            if self.e_best is None or self.x_best is None:
                self.e_best = self.e_current.clone()
                self.x_best = self.x_current.clone()

    def update_current(self, e, x):
        self.e_current = e.clone()
        self.x_current = x.clone()

    def update_best(self, e, x):
        self.e_best = e.clone()
        self.x_best = x.clone()

class GSA:
    """
    Global Search Algorithm (Generalized Simulated Annealing)
    Implementing the Morkov chain for location acceptance and local search.

    Parameters:
    -----------
    acceptance: float
        Parameter for the acceptance probability. Default value is -5.0.
        The larger the value, the more likely to accept the new location.
        With a range (-1e4,-5.0].
    visit_dist: VisitDistribution
        The visit distribution for the annealing process.
    func: function
        The objective function.
    adam_opt: AdamOptimizer
        The optimizer for the local search.
    energy_state: EnergyState
        Instance of `EnergyState` class.
    """
    def __init__(self,acceptance,visit_dist,func,adam_opt,energy_state):
        self.acceptance = acceptance
        self.visit_dist = visit_dist
        self.func = func
        self.adam_opt = adam_opt
        self.energy_state = energy_state
        self.emin = energy_state.e_current
        self.xmin = energy_state.x_current
        self.not_improved_cnt = torch.zeros(energy_state.x_current.size(0))
        self.not_imporved_max = 1000
        self.rand_gen = torch.rand
        self.temperature_step = 0
        self.K = 100*energy_state.x_current.size(1)
        self.energy_state_improved = None
        self.history = None

    def accept_decision(self, j, e, x_visit):
        r = self.rand_gen(1).item()
        p_qv_temp = 1.0-((1-self.acceptance)*(e-self.energy_state.e_current)/self.temperature_step)
        p_qv = torch.where(p_qv_temp<=0.,0,torch.exp(torch.log(p_qv_temp)/(1-self.acceptance)))
        accept = ((p_qv>=r) & (e<self.energy_state.e_current))[:,0]
        reject = ~accept
        e_update = self.energy_state.e_current.clone()
        x_update = self.energy_state.x_current.clone()
        if torch.any(accept):
            e_update[accept] = e[accept].float()
            x_update[accept] = x_visit[accept].float()
            self.energy_state.update_current(e_update,x_update)
            self.xmin = self.energy_state.x_current.clone()
        # when don not improve for a long time, force to do accept
        if torch.any(self.not_improved_cnt > self.not_imporved_max):
            too_long_not_improved = self.not_improved_cnt > self.not_imporved_max
            if j==0 or torch.any(self.energy_state.e_current < self.emin):
                self.emin[too_long_not_improved] = self.energy_state.e_current[too_long_not_improved].clone()
                self.xmin[too_long_not_improved] = self.energy_state.x_current[too_long_not_improved].clone()

    def run(self, step, temperature, **kwargs):
        self.temperature_step = temperature/float(step+1)
        self.not_improved_cnt += 1
        record_choice = kwargs.get('record_choice',None)
        if self.history is None:
            self.history = dict(
                emin_history=[self.energy_state.e_best.min().item()],
                T_list=[temperature],
                )
            if record_choice is not None:
                self.history['record_choice']['e_history'] = [self.energy_state.e_current[record_choice,0].item()]
                self.history['record_choice']['x_history'] = [self.energy_state.x_current[record_choice].detach().cpu().numpy()]
        for j in range(self.energy_state.x_current.size(1)*2):
            if j==0:
                if step==0:
                    self.energy_state_improved = torch.ones(self.energy_state.x_current.size(0)).bool()
                else:
                    self.energy_state_improved = torch.zeros(self.energy_state.x_current.size(0)).bool()
            x_visit = self.visit_dist.visiting(self.energy_state.x_current,j,temperature)
            x_update = self.energy_state.x_current.clone()
            e_update = self.energy_state.e_current.clone()
            e = self.func(x_visit,**kwargs)
            if torch.any(e<self.energy_state.e_current):
                # reach a better location
                is_better = (e<self.energy_state.e_current)[:,0]
                x_update[is_better] = x_visit[is_better].float()
                e_update[is_better] = e[is_better].float()
                self.energy_state.update_current(e_update,x_update)
                if torch.any(e<self.energy_state.e_best):
                    is_better = (e<self.energy_state.e_best)[:,0]
                    x_update[is_better] = x_visit[is_better].float()
                    e_update[is_better] = e[is_better].float()
                    self.energy_state.update_best(e_update,x_update)
                    self.energy_state_improved[is_better] = True
                    self.not_improved_cnt[is_better] = 0
            if torch.any(e>=self.energy_state.e_current):
                # do not reach a better location
                self.accept_decision(j,e,x_visit)
        self.history['emin_history'].append(self.energy_state.e_best.min().item())
        self.history['T_list'].append(temperature)
        if record_choice is not None:
            self.history['record_choice']['e_history'].append(self.energy_state.e_best[record_choice,0].item())
            self.history['record_choice']['x_history'].append(self.energy_state.x_best[record_choice].detach().cpu().numpy())

    def local_search(self, **kwargs):
        """
        Local search for the best location.
        Based on the GSA.run() results.
        If the energy state is improved, do the local search.
        Or if the energy state is not improved for a long time, do the local search.
        """
        if torch.any(self.energy_state_improved):
            # Global search is improved, do the local search
            x_update = self.energy_state.x_best.clone()
            e_update = self.energy_state.e_best.clone()
            e, x = self.adam_opt.local_search(self.energy_state.x_best, self.energy_state.e_best, **kwargs)
            if torch.any(e < self.energy_state.e_best):
                is_better = (e<self.energy_state.e_best)[:,0]
                x_update[is_better] = x[is_better]
                e_update[is_better] = e[is_better]
                self.energy_state.update_best(e_update,x_update)
                self.energy_state.update_current(e_update,x_update)
                self.not_improved_cnt[is_better] = 0
        # Check probability of a need to perform LS even if not improved
        do_ls = torch.zeros(self.energy_state.x_current.size(0)).bool()
        if self.K < 90*self.energy_state.x_current.size(1):
            pls = torch.exp(self.K*(self.energy_state.e_best-self.energy_state.e_current)/self.temperature_step)
            do_ls = self.rand_gen() <= pls
        # a long time not improved
        too_long_not_improved = self.not_improved_cnt >= self.not_imporved_max
        if torch.any(too_long_not_improved):
            do_ls[too_long_not_improved] = True
        if torch.any(do_ls):
            e, x = self.adam_opt.local_search(self.xmin, self.emin, do_ls, **kwargs)
            self.xmin = x.clone()
            self.emin = e.clone()
            self.not_improved_cnt[do_ls] = 0
            self.not_imporved_max = self.energy_state.x_current.size(1)
            if torch.any(e<self.energy_state.e_best):
                x_update = self.energy_state.x_best.clone()
                e_update = self.energy_state.e_best.clone()
                is_better = (e<self.energy_state.e_best)[:,0]
                x_update[is_better] = x[is_better]
                e_update[is_better] = e[is_better]
                self.energy_state.update_best(e_update,x_update)
                self.energy_state.update_current(e_update,x_update)
        
class AdamLocalSearch:
    """
    Local search for the best location.
    Based on the Adam optimizer.
    """
    LS_MAXITER_RATIO = 6
    LS_MAXITER_MIN = 100
    LS_MAXITER_MAX = 1000

    def __init__(self,search_bounds,func,*args,**kwargs):
        self.func = func
        self.kwargs = kwargs
        self.kwargs.pop('args',None)
        self.adam_config = self.kwargs.pop('adam',dict())
        bound_list = list(zip(*search_bounds))
        self.lower = torch.tensor(bound_list[0]).unsqueeze(0)
        self.upper = torch.tensor(bound_list[1]).unsqueeze(0)
        self.minimizer = BatchAdam(self.lower,self.upper,self.func,**self.adam_config)
        self.dim = len(bound_list[0])

        ls_max_iter = kwargs.get(
            'ls_max_iter',
            min(max(self.LS_MAXITER_RATIO*self.dim,self.LS_MAXITER_MIN),self.LS_MAXITER_MAX)
        )
        self.ls_max_iter = ls_max_iter

    def local_search(self,x,e,do_ls=None,**kwargs):
        """
        Local search from the given x location with adam optimizer.
        """
        if do_ls is None:
            do_ls = torch.ones(x.size(0)).bool()
        if torch.any(do_ls):
            x_init = x.clone()
            e_init = e.clone()
            x_init, e_init = self.minimizer(x_init, e_init, self.ls_max_iter)
            x[do_ls] = x_init[do_ls]  # Reshape to match the expected dimensions
            e[do_ls] = e_init[do_ls]  # Reshape to match the expected dimensions
        return e, x

class BatchAdam:
    """
    Parallell do adam optimization for each batch.
    """
    def __init__(self,lower,upper,func,**kwargs):
        self.lower = lower
        self.upper = upper
        self.func = func
        self.lr = kwargs.get('learning_rate', 1.e-3)
        self.betas = kwargs.get('betas', (0.9, 0.999))
        self.eps = kwargs.get('eps', 1.e-8)
        self.device = kwargs.get('device',None)
        self.max_batches = kwargs.get('max_batches',100000)
        self.gamma = kwargs.get('gamma',0.95)
        self.decay_step = kwargs.get('decay_step',10)

    def adam_step(self,g,mo,vo,t):
        beta1,beta2 = self.betas
        mt = beta1*mo+(1-beta1)*g
        vt = beta2*vo+(1-beta2)*g**2
        mt_hat = mt/(1-beta1**t)
        vt_hat = vt/(1-beta2**t)
        ret = mt_hat/(vt_hat**0.5+self.eps)
        return ret,mt,vt
    
    def __call__(self,x,e,max_iter,**kwargs):
        if self.device is None:
            self.device = x.device
        n_batch = x.size(0)
        xt = x.clone().to(self.device)
        x_best = x.clone().to(self.device)
        e_best = e.clone().to(self.device)
        mo = torch.zeros_like(x).to(self.device)
        vo = torch.zeros_like(x).to(self.device)
        lr_t = self.lr
        if n_batch<=self.max_batches:
            for t in range(1,max_iter+1):
                xt.requires_grad = True
                et = self.func(xt,**kwargs)
                gt = torch.autograd.grad(et,xt,grad_outputs=torch.ones_like(et),create_graph=False,retain_graph=False)[0]
                ga,mo,vo = self.adam_step(gt,mo,vo,t)
                xt = (xt-lr_t*ga).detach()
                et = self.func(xt,**kwargs)
                e_best = torch.where(et<e_best,et,e_best)
                x_best = torch.where(et<e_best,xt,x_best)
                if t%self.decay_step==0:
                    lr_t *= self.gamma
        else:
            x_best_list = []
            e_best_list = []
            for xb in torch.split(x,self.max_batches):
                xt = xb.clone().to(self.device)
                mo = torch.zeros_like(xb).to(self.device)
                vo = torch.zeros_like(xb).to(self.device)
                for t in range(1,max_iter+1):
                    et = self.func(xt,**kwargs)
                    gt = torch.autograd.grad(et,xt,grad_outputs=torch.ones_like(et),create_graph=False,retain_graph=False)[0]
                    ga,mo,vo = self.adam_step(gt,mo,vo,t)
                    xt = xt-lr_t*ga
                    et = self.func(xt,**kwargs)
                    eb_best = torch.where(et<e_best,et,e_best)
                    xb_best = torch.where(et<e_best,xt,x_best)
                    if t%self.decay_step==0:
                        lr_t *= self.gamma
                x_best_list.append(xb_best)
                e_best_list.append(eb_best)
            x_best = torch.cat(x_best_list,dim=0)
            e_best = torch.cat(e_best_list,dim=0)
        return x_best,e_best

        
def DualAnnealing(func,bounds,args=(),maxiter=1000,adam=dict(),initial_temp=5230.,
                  visit=2.62,accept=-5.0,no_local_search=False,x0=None,**kwargs):
    """
    Global Search Algorithm (Generalized Simulated Annealing)

    Parameters:
    -----------
    func: function
        The objective function.
    bounds: tuple
        The bounds of the parameter space.
    args: tuple,list,array_like
        The arguments of the objective function.
    maxiter: int
        The maximum number of iterations.
    adam: dict
        The arguments of the Adam optimizer.
    initial_temp: float
        The initial temperature. Default value is 5230.
    visit: float
        The visiting parameter. Default value is 2.62.
    accept: float
        The acceptance parameter. Default value is -5.0.
    no_local_search: bool
        Whether to perform local search. Default value is False.
    x0: array_like
        The given initial location. Not required.
    kwargs: dict
        The keyword arguments of the objective function.
    """
    bound_list = list(zip(*bounds))
    lower,upper = torch.tensor(bound_list[0]).unsqueeze(0),torch.tensor(bound_list[1]).unsqueeze(0)
    if (torch.any(torch.isnan(lower)) | torch.any(torch.isnan(upper))) | (torch.any(torch.isinf(lower)) | torch.any(torch.isinf(upper))):
        raise ValueError('The bounds contain NaN or inf values.')
    if (torch.any(lower>=upper)):
        raise ValueError('The lower bounds must be less than the upper bounds.')
    if not len(lower)==len(upper):
        raise ValueError('The lower and upper bounds must have the same dimension.')
    adam_optimizer = AdamLocalSearch(bounds,func,**adam)
    energy_state = EnergyState(lower,upper)
    energy_state.reset(func,x0=x0,**kwargs)

    visit_dist = VisitDistribution(lower,upper,visit)
    gsa = GSA(accept,visit_dist,func,adam_optimizer,energy_state)
    need_to_stop = False
    iteration = 0
    message = []
    optimize_res = AnnealingResult()
    optimize_res.success = True
    optimize_res.status = 0
    optimize_res.x0 = energy_state.x_current
    optimize_res.e0 = energy_state.e_current
    optimize_res.T0 = initial_temp
    t1 = np.exp((visit-1.0)*np.log(2.0))-1.0
    # Run the annealing process
    while not need_to_stop:
        for i in range(maxiter):
            # Compute step temperature
            s = float(i+2)
            t2 = np.exp((visit-1.0)*np.log(s))-1.0
            temperature = initial_temp*t1/t2
            if iteration >= maxiter:
                message.append('Maximum number of iterations reached.')
                need_to_stop = True
                break
            # Run the annealing process
            gsa.run(i,temperature,**kwargs)
            # Perform local search
            if not no_local_search:
                gsa.local_search(**kwargs)
            iteration += 1
                
    if message:
        optimize_res.success = False
        optimize_res.message = message
    else:
        optimize_res.success = True
        optimize_res.message = 'Optimization terminated successfully.'
    optimize_res.x = energy_state.x_best
    optimize_res.e = energy_state.e_best
    optimize_res.nit = iteration
    optimize_res.Tf = initial_temp*t1/(np.exp((visit-1.0)*np.log(iteration+1))-1.0)
    optimize_res.history = gsa.history
    return optimize_res

class AnnealingResult:
    """
    Result of the annealing process.
    """
    def __init__(self):
        self.success = None
        self.message = None
        self.x = None
        self.e = None
        self.nit = None
        self.status = None
        self.x0 = None
        self.e0 = None
        self.err = None
        self.T0 = None
        self.Tf = None
        self.history = None