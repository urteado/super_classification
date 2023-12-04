import numpy as np
import pandas as pd
from numba import jit, njit
import scipy.stats as stats
from sklearn.metrics import zero_one_loss
from sklearn.linear_model import ElasticNet, Ridge, LogisticRegression


## EXPERIMENT FUNCTIONS

def gendeltas(sample, var_distr, var_distr_param):
    if var_distr=='foldedgaussian': deltas=np.abs(np.random.randn(sample)) ##Folded gaussian
    if var_distr=='constant': deltas=np.ones(sample)*var_distr_param[0]
    if var_distr=='two':
        delta1 = 1
        delta2 = 2
        prob_delta1 = 0.5
        deltas = np.zeros(sample)
        for i in range(sample):                     # optimise to array
            k=np.random.uniform(0,1)
            if k<prob_delta1: deltas[i]=delta1
            else: deltas[i]=delta2
    elif var_distr=='inversegamma':
        param_a=var_distr_param[0]
        param_c=var_distr_param[1]
        deltas=stats.invgamma.rvs(param_a, scale=param_c, size=sample) 
    return deltas

@njit
def get_mean_superstats(d):
    mean_w_star = np.random.normal(0, 1, d) / np.sqrt(d)    # mean
    return mean_w_star

@njit
def get_samples_superstats(*, n, mean, prob, deltas, d, random_labels, ratio_random):
    # y = np.random.choice(np.array([1,-1]), n, np.array([prob, 1-prob])) # randomly choose 1 or -1 w.p. prob (rho) # numba likes loops better
    # X = y[:, np.newaxis] * mean + np.random.normal(0, 1, (n, d)) * np.sqrt(deltas)[:, np.newaxis] # numba likes loops better:

    y = np.zeros(n)
    if random_labels:
        for i in range(n*ratio_random):
            rand=np.random.uniform(0,1)
            if rand<prob: y[i]=1
            else: y[i]=-1
    else:
        for i in range(n):  
            rand=np.random.uniform(0,1)
            if rand<prob: y[i]=1
            else: y[i]=-1

    X = np.zeros((n,d))
    for i in range(0,n):
        X[i,:] = y[i] * mean.reshape(1,d) + np.random.normal(0,1,(1,d))*np.sqrt(deltas[i]) 
    
    return X/np.sqrt(d), y # normalise data matrix by 1/sqrt(d)

def logistic_loss(z):
    return np.log(1+np.exp(-z))


# pseudo-inverse solution for ridge regression
@njit
def ridge_estimator(X, Y, lamb):
    n, d = X.shape
    if n >= d:
        W = np.linalg.inv(X.T @ X + lamb*np.identity(d)) @ X.T @ Y.T
    elif n < d:
        W = X.T @ np.linalg.inv(X @ X.T + lamb*np.identity(n)) @ Y.T
    return W.T


def get_error_superstats(samples, p, lamb, seeds, var_distr, var_distr_param, penalty, loss, d, random_labels, ratio_random):
    eg, et, el, q_tab, m_tab, b_tab = [], [], [], [], [], []
    for i in range(seeds):

        mean = get_mean_superstats(d)                           # generate new mean at each seed
        delt=gendeltas(samples, var_distr, var_distr_param)     # generate a new set of deltas at each seed

        X_train, y_train = get_samples_superstats(n=samples, 
                                       mean=mean,
                                       prob=p,
                                       deltas=delt,
                                       d=d,
                                       random_labels=random_labels,
                                       ratio_random=ratio_random)
        X_test, y_test = get_samples_superstats(n=samples,
                                     mean=mean,
                                     prob=p,
                                     deltas=delt,
                                     d=d,
                                     random_labels=random_labels,
                                     ratio_random=ratio_random)

        if(loss == 'square'):
            X_train = np.hstack((X_train,np.ones((X_train.shape[0],1))))
            X_test = np.hstack((X_test,np.ones((X_test.shape[0],1))))
            U = ridge_estimator(X=X_train, Y=y_train, lamb=lamb)
            # built-in regressors also work but custom is best

        elif(loss=='logistic'):
            clf = LogisticRegression(penalty=penalty, solver='liblinear', fit_intercept=True, C = lamb**(-1), 
                                     max_iter=1e4, tol=1e-7, verbose=0).fit(X_train, y_train)
            w, b = clf.coef_[0], clf.intercept_[0]  

        # estimate the label
        if loss=='square':
            w=[]
            Y_train_hat = U @ X_train.T
            Y_test_hat = U @ X_test.T
            w=U[:d]
            if random_labels:
                # for random labels take the MSE of the test label and the preactivation instead of test error (which is 1/2 trivially)
                test_error = np.mean((Y_test_hat - y_test)**2)
                train_error = np.mean(( np.sign(Y_train_hat) - y_train)**2)/2
                train_loss = np.mean((Y_train_hat - y_train)**2)/2
            else:
                test_error = zero_one_loss(y_test, np.sign(Y_test_hat))
                train_error = zero_one_loss(y_train, np.sign(Y_train_hat))
                train_loss = np.mean((Y_train_hat-y_train)**2)/2
            if p==0.5: b=0 # for binary balanced classification bias is trivially 0

        elif loss=='logistic':
            train_hat = clf.predict(X_train)
            test_hat = clf.predict(X_test)
            test_error = np.mean(1-(y_test==test_hat))
            train_error = np.mean(1-(y_train==train_hat))
            train_loss = np.mean(logistic_loss( (X_train @ w) * y_train)) 

        q = w.dot(w) / d                        
        m = mean.dot(w) / np.sqrt(d)            
        eg.append(test_error)
        et.append(train_error)
        el.append(train_loss)
        q_tab.append(q) 
        m_tab.append(m) 
        b_tab.append(b) 
    return (np.mean(et), np.std(et), np.mean(eg), np.std(eg), np.mean(el), np.std(el),
            np.mean(q_tab), np.std(q_tab), 
            np.mean(m_tab), np.std(m_tab),
            np.mean(b_tab), np.std(b_tab)
           )


def simulate_superstats(prob, sc_range, lamb, variances_distr, variances_distr_param, d, seeds, penalty, loss, random_labels, ratio_random, save, verbose, data_dir):
    data = {'test_error': [], 'train_error': [], 'train_loss': [], 'test_error_std': [], 
            'train_error_std': [], 'train_loss_std': [], 'lambda': [], 'probability': [],
            'sample_complexity': [], 'samples': [], 'penalty': [], 'd': [],
            'q': [], 'q_std': [], 'm': [], 'm_std': [], 'b': [], 'b_std': []}
    
    for alpha in sc_range:
        if verbose: print('Simulating sample complexity: {}'.format(alpha))
        samples = int(alpha * d)

        et, et_std, eg, eg_std, el, el_std, q, q_std, m, m_std, b, b_std = get_error_superstats(samples=samples, 
                                                                p=prob, 
                                                                lamb=lamb, 
                                                                seeds=seeds, 
                                                                var_distr=variances_distr, 
                                                                var_distr_param=variances_distr_param,
                                                                penalty=penalty,
                                                                loss=loss,
                                                                d=d,
                                                                random_labels=random_labels,
                                                                ratio_random=ratio_random)
        if verbose: print('gen error: {}, train error: {}, train loss: {} '.format(eg, et, el))

        data['sample_complexity'].append(alpha)
        data['samples'].append(samples)
        data['probability'].append(prob)
        data['penalty'].append(penalty)
        data['lambda'].append(lamb)
        data['d'].append(d) 
        data['test_error'].append(eg)
        data['test_error_std'].append(eg_std)
        data['train_error'].append(et)
        data['train_error_std'].append(et_std)
        data['train_loss'].append(el)
        data['train_loss_std'].append(el_std)
        data['q'].append(q)
        data['q_std'].append(q_std)
        data['m'].append(m)
        data['m_std'].append(m_std)
        data['b'].append(b)
        data['b_std'].append(b_std)

    df_data = pd.DataFrame.from_dict(data)
    if save: 
        df_data.to_csv(data_dir+'sim_sc'+str(sc_range[0])+'_to_'+str(sc_range[-1])+'_d'+str(d)+'_rho'+str(prob)+'_lamb'+str(lamb)+'_randomlabels'+str(random_labels)+'.csv')
    return df_data


