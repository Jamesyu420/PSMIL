import numpy as np
import time
import scipy.stats as stats

def L2norm(x1, x2): # Compute L2 distance
    return np.sum((x1 - x2) ** 2) 

## Instance-based MLE
def InsMLE(X, A, Y):
    '''
    Input: 
    X: Features, Y: Bag-level labels
    A: Instance-level labels
    '''
    N, M, p = X.shape # N: sample size for bags, M: sample size for instances, p: feature dimension
    # IMLE for alpha, pi, mu1 and mu0
    alpha = np.sum(Y) / N 
    A_sum = np.sum(A)
    pi_hat = A_sum / (M * np.sum(Y)) 
    mu1 = np.sum(A[:, :, np.newaxis] * X, axis=(0, 1)) / A_sum 
    mu0 = np.sum((1 - A)[:, :, np.newaxis] * X, axis=(0, 1)) / (M * N - A_sum) 

    # Reshape mu0 and mu1 for broadcasting
    mu0_reshaped = mu0.reshape(1, 1, p)
    mu1_reshaped = mu1.reshape(1, 1, p)

    # Compute differences
    diff_mu0 = X - mu0_reshaped
    diff_mu1 = X - mu1_reshaped

    # Vectorize the calculation of Sigma
    Sigma = np.einsum('ijk,ijl->kl', (1 - A)[:, :, np.newaxis] * diff_mu0, diff_mu0) \
          + np.einsum('ijk,ijl->kl', A[:, :, np.newaxis] * diff_mu1, diff_mu1)
    Sigma /= (M * N)

    return {
        'mu1': mu1,
        'mu0': mu0,
        'alpha': alpha,
        'pi': pi_hat,
        'Sigma': Sigma
    }

## EM algorithm for bag-based MLE
def BagMLE(X, Y, mu1_ground, mu0_ground, Sigma_ground, pi_ground, prt=False, iter=100, tol=1e-10):
    '''
    Input: 
    X: Features, Y: Bag-level labels
    mu1_ground: Initial mean vector for class 1
    mu0_ground: Initial mean vector for class 0
    Sigma_ground: Initial covariance matrix
    pi_ground: Initial mixing probability for positive instances on a positive bag
    '''
    N, M, p = X.shape # N: sample size for bags, M: sample size for instances, p: feature dimension
    # Initialization for the estimators
    pi = pi_ground
    mu1 = np.array(mu1_ground)
    mu0 = np.array(mu0_ground)
    Sigma = Sigma_ground
    alpha = np.mean(Y)
    loglik = 0
    # Create copies for storing the new iteration estimators
    mu1_new = mu1.copy()
    mu0_new = mu0.copy()
    Sigma_new = Sigma.copy()
    pi_new = pi_ground
    # Conduct EM algorithm for t-step iteration
    for t in range(iter):
        t1 = time.time()

        if t == 0:
            ## E-step: Compute the posterior probability
            Omega = np.linalg.inv(Sigma) # Compute the precision matrix
            aa = (mu0.T@Omega@mu0-mu1.T@Omega@mu1)/2+np.log(pi_new/(1-pi_new))
            delta = Omega@(mu1-mu0)
            pi_im = 1 - 1/(1+np.exp(X@delta+aa)) 
            
            eps = np.finfo(np.float64).eps
            pi_im[pi_im < eps] = eps
            pi_im[pi_im > 1 - eps] = 1 - eps

        ## M-step: Compute the t-th step estimator
        Y_reshape = Y[:, np.newaxis]
        Y_pi_im = Y_reshape * pi_im
        pi_new = np.sum(Y_pi_im) / (M * np.sum(Y))
        
        # Vectorized computation of mu1_new and mu0_new
        mu1_new = np.sum(X * Y_pi_im[:, :, np.newaxis], axis=(0, 1)) / np.sum(Y_pi_im)
        mu0_new = np.sum(X * (1 - Y_pi_im)[:, :, np.newaxis], axis=(0, 1)) / np.sum(1 - Y_pi_im)

        # Reshape and broadcast Y and pi_im for Sigma_new computation
        Y_expanded = Y[:, np.newaxis, np.newaxis]
        pi_im_expanded = pi_im[:, :, np.newaxis]

        # Compute X - mu1_new and X - mu0_new
        diff_mu1_new = X - mu1_new
        diff_mu0_new = X - mu0_new

        # Vectorize the calculation of Sigma_new
        Sigma_new = np.einsum('ijk,ijl->kl', Y_expanded * pi_im_expanded * diff_mu1_new, diff_mu1_new) \
                  + np.einsum('ijk,ijl->kl', (1 - Y_expanded * pi_im_expanded) * diff_mu0_new, diff_mu0_new)

        Sigma_new /= (N * M)

        ## E-step: Compute the posterior probability
        Omega = np.linalg.inv(Sigma_new)
        aa = (mu0_new.T@Omega@mu0_new-mu1_new.T@Omega@mu1_new)/2+np.log(pi_new/(1-pi_new))
        delta = Omega@(mu1_new-mu0_new)
        pi_im = 1 - 1/(1+np.exp(X@delta+aa))

        t2 = time.time()
        # Compute the distance between the t-th step estimator and the (t+1)-th step estimator
        dist = L2norm(mu1, mu1_new) +L2norm(mu0, mu0_new) + L2norm(Sigma, Sigma_new) + L2norm(pi, pi_new)
        if prt:
            if dist < tol:
                print(f'FINISH:{t}-{np.round(t2-t1,2)}s: mu1={L2norm(mu1, mu1_new)}, mu0={L2norm(mu0, mu0_new)}, sig={L2norm(Sigma, Sigma_new)}')
                break
            else:
                print(f'{t}-{np.round(t2-t1,2)}s: mu1={L2norm(mu1, mu1_new)}, mu0={L2norm(mu0, mu0_new)}, sig={L2norm(Sigma, Sigma_new)}')
            print(pi_new)
        else:
            if dist < tol: break
            
        mu1 = mu1_new.copy()
        mu0 = mu0_new.copy()
        Sigma = Sigma_new.copy()
        pi = pi_new

    return {
        'mu1': mu1,
        'mu0': mu0,
        'alpha': alpha,
        'pi': pi,
        'Sigma': Sigma,
        'pi_im': pi_im,
        'iter': t
    }

## EM algorithm for Subsampling-based MLE
def SubMLE(X, Y, A, mu1_ground, mu0_ground, Sigma_ground, pi_ground, est, 
            alpha_n=-4.5, prt=False, iter=100, tol=1e-10, seed=0):
    '''
    Input: 
    X: Features, Y: Bag-level labels
    mu1_ground: Initial mean vector for class 1
    mu0_ground: Initial mean vector for class 0
    Sigma_ground: Initial covariance matrix
    pi_ground: Initial mixing probability for positive instances on a positive bag
    est: Bag-based MLE
    alpha_n: alpha_n is used to control the overall subsampling fraction
    '''
    N, M, p = X.shape # N: sample size for bags, M: sample size for instances, p: feature dimension
    np.random.seed(seed)
    ## Generate the subsampling indicators for instances based on BMLE
    mu1_hat, mu0_hat, Sigma_hat, pi_hat = est['mu1'], est['mu0'], est['Sigma'], est['pi']
    
    Sigma_hat = Sigma_hat + 1e-10 * np.eye(Sigma_hat.shape[0])
    beta_hat = np.linalg.solve(Sigma_hat, mu1_hat - mu0_hat) # Estimate beta by BMLE
    # Compute the subsampling probabilities
    gamma_im = np.exp(alpha_n + np.sum(X * beta_hat, axis=2))
    # gamma_im = gamma_im / (1 + gamma_im)
    # gamma = np.sum(gamma_im) / (M * N)
    gamma_im = gamma_im / (1 + gamma_im) * Y[:, np.newaxis]
    gamma = np.sum(gamma_im) / (M * np.sum(Y))
    

    # Compute the subsampling instances
    s_im = np.random.binomial(1, gamma_im.flatten()).reshape(N, M)
    s_a1 = s_im * A 
    # Initialization for the estimators
    alpha = np.mean(Y)
    pi_hat = pi_ground
    mu1 = mu1_ground
    mu0 = mu0_ground
    Sigma = Sigma_ground
    
    # Create copies for storing the new iteration estimators
    mu1_new, mu0_new, Sigma_new, pi_new = mu1.copy(), mu0.copy(), Sigma.copy(), pi_hat
    Y_reshape = Y[:, np.newaxis]
    Y_expanded = Y[:, np.newaxis, np.newaxis]
    
    eps = np.finfo(np.float64).eps
    # Conduct EM algorithm for t-step iteration
    for t in range(iter):
        t1 = time.time()
        Sigma = Sigma + eps * np.eye(Sigma.shape[0])
        ## E-step: Compute the posterior probability
        Omega = np.linalg.inv(Sigma)
        aa = (mu0.T@Omega@mu0-mu1.T@Omega@mu1)/2+np.log(pi_new/(1-pi_new))
        delta = Omega@(mu1-mu0)
        pi_im = 1 - 1/(1+np.exp(X@delta+aa))
        
        pi_im[pi_im < eps] = eps
        pi_im[pi_im > 1 - eps] = 1 - eps
        
        s_pi_im = s_im * pi_im
        ## M-step: Compute the t-th step estimator
        # Compute combined term for pi_new, mu1_new, and mu0_new calculations
        combined_term = Y_expanded * (pi_im[:, :, np.newaxis] + s_a1[:, :, np.newaxis] - s_pi_im[:, :, np.newaxis])

        # Vectorized computation of pi_new
        pi_new = np.sum(combined_term) / (M * np.sum(Y))

        # Vectorized computation of mu1_new and mu0_new
        mu1_new = np.sum(combined_term * X, axis=(0, 1)) / np.sum(combined_term, axis=(0, 1))
        mu0_new = np.sum((1 - combined_term) * X, axis=(0, 1)) / np.sum(1 - combined_term, axis=(0, 1))

        # Vectorize the calculation of Sigma_new
        diff_mu1 = X - mu1_new
        diff_mu0 = X - mu0_new
        term1 = combined_term * diff_mu1
        term2 = (1 - combined_term) * diff_mu0
        Sigma_new = np.einsum('ijk,ijl->kl', term1, diff_mu1) + np.einsum('ijk,ijl->kl', term2, diff_mu0)
        Sigma_new /= (N * M)

        Sigma_new = Sigma_new + eps * np.eye(Sigma_new.shape[0])
        Omega_new = np.linalg.inv(Sigma_new)
        aa = (mu0_new.T@Omega_new@mu0_new-mu1_new.T@Omega@mu1_new)/2+np.log(pi_new/(1-pi_new))
        delta = Omega_new@(mu1_new-mu0_new)
        pi_im = 1 - 1/(1+np.exp(X@delta+aa))
        t2 = time.time()
        
        pi_im[pi_im < eps] = eps
        pi_im[pi_im > 1 - eps] = 1 - eps
        # Compute the distance between the t-th step estimator and the (t+1)-th step estimator
        dist = L2norm(mu1, mu1_new) +L2norm(mu0, mu0_new) + L2norm(Sigma, Sigma_new) + L2norm(pi_hat, pi_new)
        if prt:
            if dist < tol:
                print(f'FINISH:{t}-{np.round(t2-t1,2)}s: mu1={L2norm(mu1, mu1_new)}, mu0={L2norm(mu0, mu0_new)}, sig={L2norm(Sigma, Sigma_new)}')
                break
            else:
                print(f'{t}-{np.round(t2-t1,2)}s: mu1={L2norm(mu1, mu1_new)}, mu0={L2norm(mu0, mu0_new)}, sig={L2norm(Sigma, Sigma_new)}')
            print(pi_new)
        else:
            if dist < tol: break
        
        mu1, mu0, Sigma, pi_hat = mu1_new.copy(), mu0_new.copy(), Sigma_new.copy(), pi_new

    return {
        'mu1': mu1,
        'mu0': mu0,
        'alpha': alpha,
        'pi': pi_hat,
        'Sigma': Sigma,
        'pi_im': pi_im,
        'iter': t,
        'prop': gamma
    }


def SubMLEpure(X, Y, A, mu1_ground, mu0_ground, Sigma_ground, pi_ground, est, 
            prop, iter=100, tol=1e-10, seed=0):
    N, M, p = X.shape
    np.random.seed(seed)

    mu1_hat, mu0_hat, Sigma_hat, pi_hat = est['mu1'], est['mu0'], est['Sigma'], est['pi']
    
    Sigma_hat = Sigma_hat + 1e-10 * np.eye(Sigma_hat.shape[0])
    beta_hat = np.linalg.solve(Sigma_hat, mu1_hat - mu0_hat)
    
    gamma_im = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            gamma_im[i, j] = prop

    s_im = np.random.binomial(1, gamma_im.flatten()).reshape(N, M)
    s_a1 = s_im * A

    alpha = np.mean(Y)
    pi_hat = pi_ground
    mu1 = mu1_ground
    mu0 = mu0_ground
    Sigma = Sigma_ground
    loglik = 0

    mu1_new, mu0_new, Sigma_new, pi_new = mu1.copy(), mu0.copy(), Sigma.copy(), pi_hat
    Y_reshape = Y[:, np.newaxis]
    Y_expanded = Y[:, np.newaxis, np.newaxis]
    
    eps = np.finfo(np.float64).eps
    for t in range(iter):
        t1 = time.time()
        Sigma = Sigma + eps * np.eye(Sigma.shape[0])
        
        Omega = np.linalg.inv(Sigma)
        aa = (mu0.T@Omega@mu0-mu1.T@Omega@mu1)/2+np.log(pi_new/(1-pi_new))
        delta = Omega@(mu1-mu0)
        pi_im = 1 - 1/(1+np.exp(X@delta+aa))
        
        pi_im[pi_im < eps] = eps
        pi_im[pi_im > 1 - eps] = 1 - eps
        
        s_pi_im = s_im * pi_im
        
        # Compute combined term for pi_new, mu1_new, and mu0_new calculations
        combined_term = Y_expanded * (pi_im[:, :, np.newaxis] + s_a1[:, :, np.newaxis] - s_pi_im[:, :, np.newaxis])

        # Vectorized computation of pi_new
        pi_new = np.sum(combined_term) / (M * np.sum(Y))

        # Vectorized computation of mu1_new and mu0_new
        mu1_new = np.sum(combined_term * X, axis=(0, 1)) / np.sum(combined_term, axis=(0, 1))
        mu0_new = np.sum((1 - combined_term) * X, axis=(0, 1)) / np.sum(1 - combined_term, axis=(0, 1))

        # Vectorize the calculation of Sigma_new
        diff_mu1 = X - mu1_new
        diff_mu0 = X - mu0_new
        term1 = combined_term * diff_mu1
        term2 = (1 - combined_term) * diff_mu0
        Sigma_new = np.einsum('ijk,ijl->kl', term1, diff_mu1) + np.einsum('ijk,ijl->kl', term2, diff_mu0)
        Sigma_new /= (N * M)

        Sigma_new = Sigma_new + eps * np.eye(Sigma_new.shape[0])
        Omega_new = np.linalg.inv(Sigma_new)
        aa = (mu0_new.T@Omega_new@mu0_new-mu1_new.T@Omega@mu1_new)/2+np.log(pi_new/(1-pi_new))
        delta = Omega_new@(mu1_new-mu0_new)
        pi_im = 1 - 1/(1+np.exp(X@delta+aa))
        t2 = time.time()
        
        pi_im[pi_im < eps] = eps
        pi_im[pi_im > 1 - eps] = 1 - eps
        
        dist = L2norm(mu1, mu1_new) +L2norm(mu0, mu0_new) + L2norm(Sigma, Sigma_new) + L2norm(pi_hat, pi_new)
        if dist < tol:
            break

        mu1, mu0, Sigma, pi_hat = mu1_new.copy(), mu0_new.copy(), Sigma_new.copy(), pi_new

    return {
        'mu1': mu1,
        'mu0': mu0,
        'alpha': alpha,
        'pi': pi_hat,
        'Sigma': Sigma,
        'pi_im': pi_im,
        'iter': t,
        'prop': prop
    }

## Compute alpha_n based on gamma
def sub_alphan(X, Y, est, gamma_target, low=-100, high=100, prt=False, iter=100):
    '''
    Input: 
    X: Features, Y: Bag-level labels
    est: Bag-based MLE
    gamma_target: the target subsampling fraction
    '''
    t1 = time.time()
    N, M, p = X.shape # N: sample size for bags, M: sample size for instances, p: feature dimension
    ## Generate the subsampling indicators for instances based on BMLE
    mu1_hat, mu0_hat, Sigma_hat, pi_hat = est['mu1'], est['mu0'], est['Sigma'], est['pi']
    Sigma_hat = Sigma_hat + 1e-10 * np.eye(Sigma_hat.shape[0])
    beta_hat = np.linalg.solve(Sigma_hat, mu1_hat - mu0_hat) # Estimate beta by BMLE
    # Compute the subsampling probabilities
    Xbeta = np.sum(X * beta_hat, axis=2)

    # Binary search to solve for alpha_n
#     low, high = -100, 100  # Search range for alpha_n
    tol_bisect = 1e-5    # Tolerance for the binary search convergence
    max_iter_bisect = 100

    for _ in range(max_iter_bisect):
        mid = (low + high) / 2
        gamma_im = np.exp(mid + Xbeta)
        gamma_im = gamma_im / (1 + gamma_im) * Y[:, np.newaxis]
        gamma_mid = np.sum(gamma_im) / (M * np.sum(Y))

        if abs(gamma_mid - gamma_target) < tol_bisect:
            alpha_n = mid
            break
        elif gamma_mid < gamma_target:
            low = mid
        else:
            high = mid
    else:
        alpha_n = (low + high) / 2  # Finally take the midpoint
    t2 = time.time()
    if prt:
        print(f'Solved alpha_n = {alpha_n:.4f} for gamma_target = {gamma_target}')    
        print(f'Compute alpha_n based on gamma: {np.round(t2-t1,4)}s')
    return alpha_n

