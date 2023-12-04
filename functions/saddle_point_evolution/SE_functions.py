import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import erfc
from scipy.integrate import quad
from numba import jit, njit, prange  # if not using, also comment out the @jit/@njit above functions
from scipy.stats import invgamma


S = 100000 # global variable for MC averaging 

# utils functions

@njit
def Q_arr(x):
    result=np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        result[i]=(1/2) * erfc(x[i]/np.sqrt(2))
    return result # checked OK


# VARIANCE DISTRIBUTION

def gendeltas(sample, var_distr, var_distr_param):
    if var_distr=='constant': deltas=np.ones(sample)
    if var_distr=='inversegamma':
        param_a=var_distr_param[0]
        param_c=var_distr_param[1]
        deltas=stats.invgamma.rvs(param_a, scale=param_c, size=sample)
    # deltas = 1 / np.random.gamma(shape=param_a, scale=1/(param_a-1), size=sample)
    if var_distr=='foldedgaussian': deltas=np.abs(np.random.randn(sample)) ##Folded gaussian
    return deltas


# function for infinite-variance interpolation
def gendeltas_interp(sample, ratio_infvar):
    deltas=np.ones(sample)
    cutoff=int(sample*ratio_infvar) # cutoff 
    param_a=0.5
    deltas[:cutoff]=stats.invgamma.rvs(param_a, scale=1, size=cutoff)
    return deltas


def integrate_invgamma(a, c, fun):
    if a == 'inf':
        return fun(1)
    else:
        rv = invgamma(a, scale=c)
        return quad(lambda x: fun(x)*rv.pdf(x), 0, 30, limit=70)[0]


# ORDER PARAMETER UPDATE FUNCTIONS

def findv(lam,alpha,var_distr,var_distr_param,int_type):
    v, vhat = 100, 100
    eps=100
    a=var_distr_param[0]
    c=var_distr_param[1]
    while (eps>1e-5):
        vold=v
        vhatold=vhat
        if int_type=='MC':
            delta=gendeltas(S,var_distr,var_distr_param)  # MC approach
            vhat=alpha*(delta*1./(1+v*delta)).mean()      # MC approach
        elif int_type=='quad':
            vhat = alpha*integrate_invgamma(a, c, lambda delta: delta*1./(1+v*delta))    # quad integration
        v=1./(lam+vhat)
        eps=np.abs(v-vold)+np.abs(vhat-vhatold)
    return v, vhat


def update_qhat(v,q,m,b,rho,alpha,var_distr,var_distr_param,int_type):
    Y=np.array([1,-1])
    if int_type=='MC':
        deltas=gendeltas(S,var_distr,var_distr_param)     # MC approach
        ED1=(deltas/((1+v*deltas)**2)).mean()
        deltas1=gendeltas(S,var_distr,var_distr_param)
        ED2=(deltas1**2/(1+v*deltas1)**2).mean()
    elif int_type=='quad':
        a=var_distr_param[0]
        c=var_distr_param[1]
        ED1=integrate_invgamma(a, c, lambda delta: delta/((1+v*delta)**2)) 
        ED2=integrate_invgamma(a, c, lambda delta: delta**2/(1+v*delta)**2) 
    qhat = (rho @ (Y-m-b)**2)*ED1+q*ED2
    qhat = alpha*qhat
    return qhat


def update_mhat(v,q,m,b,rho,alpha,var_distr,var_distr_param,int_type):
    Y=np.array([1,-1])
    if int_type=='MC':
        deltas=gendeltas(S,var_distr,var_distr_param)     # MC approach
        ED0= (1./(1+v*deltas)).mean()
    elif int_type=='quad':
        a=var_distr_param[0]
        c=var_distr_param[1]
        ED0=integrate_invgamma(a, c, lambda delta: 1./(1+v*delta)) 
    mhat = alpha*rho*(Y-m-b)*ED0
    return mhat


def update_overlaps(v,q,m,b,rho,alpha,lam,var_distr,var_distr_param,int_type):
    Y=np.array([1,-1])
    b=rho @ (Y-m)
    qhat=update_qhat(v,q,m,b,rho,alpha,var_distr,var_distr_param,int_type)
    mhat=update_mhat(v,q,m,b,rho,alpha,var_distr,var_distr_param,int_type)
    mumu=np.array([[1,-1],[-1,1]]) #Gram matrix
    q= (mhat.T @ mumu @ mhat + qhat)*v*v
    m=(np.dot(mhat.T , mumu))*v
    return v, q, m, b



# PERFORMANCE METRIC COMPUTATIONS

def get_gen_error_square(q, m, b, prob, var_distr, var_distr_param):
    if var_distr=='constant': 
        Delta=var_distr_param[0]
        gen_error = prob * np.mean(Q_arr((1-m[0] + b)/np.sqrt(q*Delta))) \
            + (1-prob) * np.mean(Q_arr((-1-m[1]-b)/np.sqrt(q*Delta))) # gaussian
    else:
        S_gen=int(5e4)
        deltas1=gendeltas(S_gen,var_distr,var_distr_param)
        deltas2=gendeltas(S_gen,var_distr,var_distr_param)
        size=int(1e4)
        gen_error = gen_error_loop(q, m, b, prob, deltas1, deltas2, size)
    
    return gen_error


@njit
def gen_error_loop(q, m, b, prob, deltas1, deltas2, size):
    xi = np.random.normal(0, 1, (len(deltas1), size))
    gen_errors=np.zeros(len(deltas1))
    for i in range(len(deltas1)):
        k1 = np.sign( xi[i]*np.sqrt(q*deltas1[i])+m[0]+b ) #yk=1
        y1 = np.ones(size)
        k2 = np.sign( xi[i]*np.sqrt(q*deltas2[i])+m[1]-b )
        y2=-y1
        k1_err = np.mean(np.equal(k1,y1))
        k2_err = np.mean(np.equal(k2,y2))
        gen_errors[i] = 1 - (prob*k1_err + (1-prob)*k2_err)
    return np.mean(gen_errors)


@njit
def get_train_err_sq(q, m, V, b, prob, deltas):
    S_gen=int(1e5)
    xi=np.random.normal(0,1,S_gen)
    omega1=np.sqrt(deltas*q)*xi+m[0]+b
    omega2=np.sqrt(deltas*q)*xi+m[1]+b
    sign1=np.zeros(S_gen)
    sign2=np.zeros(S_gen)
    for i in range(len(deltas)): 
        h1 = (omega1[i]+deltas[i]*V) /(1+V*deltas[i])
        h2 = (omega2[i]-deltas[i]*V) /(1+V*deltas[i])
        sign1[i] = np.sign(h1) #yk=1
        sign2[i] = np.sign(h2) #yk=1
    y1 = np.ones(S_gen)
    y2=-y1
    ED1=(sign1==y1).mean()
    ED2=(sign2==y2).mean()
    train_error=1-(prob*ED1+(1-prob)*ED2)
    return train_error


@njit
def get_train_loss(q, m, V, b, deltas):
    S_gen=int(1e5)
    xi=np.random.normal(0,1,S_gen)
    omega1=np.sqrt(deltas*q)*xi+m[0]+b
    omega2=np.sqrt(deltas*q)*xi+m[1]+b
    l1=np.zeros(S_gen)
    l2=np.zeros(S_gen)
    for i in range(len(deltas)): 
        h1 = (omega1[i]+deltas[i]*V) /(1+V*deltas[i])
        h2 = (omega2[i]-deltas[i]*V) /(1+V*deltas[i])
        l1[i] = (h1-1)**2/2 #yk=1
        l2[i] = (h2+1)**2/2 #yk=-1
    ED1=l1.mean()
    ED2=l2.mean()
    train_loss=ED1+ED2
    return train_loss


# MAIN UPDATE FUNCTIONS

@njit
def damp(q_new, q_old, coef_damping=0.6):
    return (1-coef_damping) * q_new + coef_damping * q_old

def iterate_sp(alpha, lamb, var_distr, var_distr_param, prob=.5, max_iter=int(500), eps=1e-9, init_condition=(0.01,0.1,0.1,0), int_type='quad', verbose=True):
    """ 
    Update state evolution equations. 
    Parameters:
    * eps = threshold to reach convergence.
    * max_iter = maximum number of steps if convergence not reached.
    """
    
    # Initialise qu and qv
    v = np.zeros(max_iter)
    q = np.zeros(max_iter)
    m = np.zeros((max_iter,2))
    b = np.zeros(max_iter)
    
    q[0], m[0,0], m[0,1], b[0] = init_condition
    v[0], _ = findv(lamb,alpha,var_distr,var_distr_param,int_type)
    if verbose: print('v: ',v[0])
    rho=np.array([prob,1-prob])
    
    if verbose: print('starting state evolution...')
    for t in range(max_iter - 1):
        v[t+1], qtmp, mtmp, btmp = update_overlaps(v[t],q[t],m[t],b[t],rho,alpha,lamb,var_distr,var_distr_param,int_type)
        q[t+1], m[t+1], b[t+1] = damp(qtmp, q[t]), damp(mtmp, m[t]), damp(btmp, b[t])
        if verbose:
            if t%50==0 and t>0: print('t: {}, v: {}, q: {}, m: {}, b: {}'.format(t, v[t+1], q[t+1], m[t+1], b[t+1]))
        diff = np.abs(q[t + 1] - q[t]) + np.linalg.norm(m[t+1]-m[t])+ np.abs(b[t+1] - b[t])
        if diff < eps:
            if verbose: print('reached precision, break evolution at t =',t)
            break 
        if t==max_iter-1 and verbose: print('reached max steps, break evolution')
    return v[:t + 1], q[:t + 1], m[:t + 1], b[:t + 1], t



def run_SE_binary(alphas, lamb, var_distr, var_distr_param, prob, max_iter=int(500), eps=1e-9, init=(0.01,0.1,0.1,0), integration_type='quad', verbose=True):
    result = {'sample_complexity': [], 'v': [], 'q': [], 'm1': [], 'm2': [], 'b': [],
                    'test_error':[], 'train_error':[], 'train_loss':[], 't': [],'lambda': []}
    for alpha in alphas:
        if verbose: print('α={}'.format(alpha))
        
        v, q, m, b, t = iterate_sp(alpha, lamb, var_distr, var_distr_param, prob, max_iter=int(500), eps=1e-9, init_condition=init, int_type=integration_type, verbose=verbose)
        
        test_error = get_gen_error_square(q[-1], m[-1], b[-1], prob, var_distr, var_distr_param)
        S_gen=int(1e5)
        if verbose: print('computing errors...')
        deltas_tre=gendeltas(S_gen,var_distr,var_distr_param)
        train_error = get_train_err_sq(q[-1], m[-1], v[-1], b[-1], prob, deltas_tre)
        deltas_trl=gendeltas(S_gen,var_distr,var_distr_param)
        train_loss = get_train_loss(q[-1], m[-1], v[-1], b[-1], deltas_trl)
        result['sample_complexity'].append(alpha)

        result['v'].append(v[-1])
        result['q'].append(q[-1])
        result['m1'].append(m[-1,0])
        result['m2'].append(m[-1,1])
        result['b'].append(b[-1])

        result['test_error'].append(test_error)
        result['train_error'].append(train_error)
        result['train_loss'].append(train_loss)
        result['t'].append(t)
        result['lambda'].append(lamb)
        if verbose: print('t: {}, v: {}, q: {}, m: {}, b: {}, generr: {} \n'.format(t, v[-1], q[-1], m[-1], b[-1], test_error))
        
        init = (q[-1], m[-1,0], m[-1,1], b[-1]) 
            
    result_df = pd.DataFrame.from_dict(result)
    return result_df 
