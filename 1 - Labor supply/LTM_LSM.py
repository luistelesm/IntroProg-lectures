import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def tax_pmt(w, l, params):
    """
    Calculate the tax payment given wage, labor supply, and tax parameters.
    
    Args:
    
        w (float): wage
        l (float): labor supply
        params (dict): dictionary of problem parameters
        
    Returns:
    
        (float): tax payment
    """
    tau_0, tau_1, k = params['t0'], params['t1'], params['k'],
    return tau_0 * w * l + tau_1 * np.fmax(w*l - k, 0)

def cons_given_budget(w, l, params):
    """
    Calculate consumption given wage, labor supply, and parameters, based on budget constraint.

    Args:
        
       w (float): wage
       l (float): labor supply
       params (dict): dictionary of problem parameters

    Returns:
            
        (float): consumption
    
    """
    m = params['m']
    return m + w * l - tax_pmt(w, l, params)

def u(w, l, params): return np.log(cons_given_budget(w, l, params)) - params['v'] * l ** (1 + 1 / params['eps']) / (1 + 1 / params['eps'])

def optimal_l(w, params):
    """
    Calculate optimal labor supply given wage and parameters.
    
    Args:
    
        w (float): wage
        params (dict): dictionary of problem parameters
        
    Returns:
    
        (float): optimal labor supply
    """
    # 1. call solver
    sol = optimize.minimize_scalar(lambda l: -u(w, l, params),
                                   method='bounded',
                                   bounds=(0, 1))
    
    # 2. unpack solution
    opt_l = sol.x

    return opt_l



