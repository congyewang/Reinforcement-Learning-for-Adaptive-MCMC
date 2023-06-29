import numpy as np
from scipy.stats import multivariate_normal

np.random.seed(1234)


def env_mh(theta_curr, cov_curr, cov_prop, log_pi):
    log_pi_curr = log_pi(theta_curr)
    theta_prop = np.random.multivariate_normal(theta_curr, cov_curr)
    log_pi_prop = log_pi(theta_prop)

    log_alpha = log_pi_prop - log_pi_curr + multivariate_normal.logpdf(theta_curr, theta_prop, cov_prop) - multivariate_normal.logpdf(theta_prop, theta_curr, cov_curr)

    if np.log(np.random.uniform()) < log_alpha:
        theta_curr = theta_prop
        accepted_status = True
    else:
        accepted_status = False

    return theta_curr, accepted_status

def policy_mh(theta_start, policy_cov, log_pi, nits):
    d = theta_start.size
    store = np.zeros((nits+1, d))
    # log_pi_theta_prop = np.zeros(nits, d)
    nacc = 0
    theta_curr = theta_start
    log_pi_curr = log_pi(theta_curr)
    store[0] = theta_curr

    for i in range(nits):
        # Current State
        sigma2_curr = policy_cov(theta_curr)

        # Proposed State
        theta_prop = np.random.multivariate_normal(theta_curr, sigma2_curr)
        sigma2_prop = policy_cov(theta_prop)
        log_pi_prop = log_pi(theta_prop)

        log_alpha = log_pi_prop - log_pi_curr + multivariate_normal.logpdf(theta_curr, theta_prop, sigma2_prop) - multivariate_normal.logpdf(theta_prop, theta_curr, sigma2_curr)

        if np.log(np.random.uniform()) < log_alpha:
            theta_curr = theta_prop
            log_pi_curr = log_pi_prop
            nacc = nacc + 1
        store[i+1] = theta_curr

    acc = nacc/nits

    return store, acc

def optimal_policy_cov(
        x,
        theta0=0,
        theta1=0,
        theta2=0,
        theta3=1,
        theta4=2.5,
        theta5=0,
        theta6=2.5,
        theta7=0,
        theta8=1,
        theta9=2.5,
        theta10=0,
        theta11=2.5,
        theta12=0):
    phi = np.arccos(theta0 + theta1*x[0] + theta2*x[1])
    alpha = theta3**2 + theta4**2 * (x[0] - theta5)**2 + theta6**2 * (x[1] - theta7)**2
    beta = theta8**2 + theta9**2 * (x[0] - theta10)**2 + theta11**2 * (x[1] - theta12)**2

    t1 = np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]])
    t2 = np.array([[alpha, 0], [0, beta]])

    sigma2 = t1 @ t2 @ t1.T

    return sigma2

def jump_distance(store):
    diffs = np.diff(store, axis=0)
    distances = np.linalg.norm(diffs, axis=1) ** 2

    return distances