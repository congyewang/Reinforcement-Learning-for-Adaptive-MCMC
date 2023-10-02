import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import roots_hermite
from functools import partial

INF = 3.4028235e+38 # Corresponds to the value of FLT_MAX in C++
SEED = 1234
generator = np.random.Generator(np.random.PCG64(SEED))


def env_mh(theta_curr, sigma_curr, policy_func, log_p):
    theta_prop = generator.normal(theta_curr, sigma_curr).flatten()
    sigma_prop = policy_func(theta_prop)

    log_p_prop = log_p(theta_prop)
    log_p_curr = log_p(theta_curr)
    log_q_prop = norm.logpdf(theta_prop, loc=theta_curr, scale=sigma_curr)
    log_q_curr = norm.logpdf(theta_curr, loc=theta_prop, scale=sigma_prop)

    log_alpha = log_p_prop \
            - log_p_curr \
            + log_q_curr \
            - log_q_prop

    if np.log(generator.uniform()) < log_alpha:
        theta_curr = theta_prop
        accepted_status = True
    else:
        accepted_status = False

    return theta_curr, accepted_status, theta_prop

def policy_mh(theta_start, policy_cov, log_p, nits):
    d = theta_start.size
    store = np.zeros((nits+1, d))
    # log_p_theta_prop = np.zeros(nits, d)
    nacc = 0
    theta_curr = theta_start
    log_p_curr = log_p(theta_curr)
    store[0] = theta_curr

    for i in range(nits):
        # Current State
        sigma2_curr = policy_cov(theta_curr)

        # Proposed State
        theta_prop = generator.multivariate_normal(theta_curr, sigma2_curr)
        sigma2_prop = policy_cov(theta_prop)
        log_p_prop = log_p(theta_prop)

        log_alpha = log_p_prop - log_p_curr + multivariate_normal.logpdf(theta_curr, theta_prop, sigma2_prop) - multivariate_normal.logpdf(theta_prop, theta_curr, sigma2_curr)

        if np.log(generator.uniform()) < log_alpha:
            theta_curr = theta_prop
            log_p_curr = log_p_prop
            nacc = nacc + 1
        store[i+1] = theta_curr

    acc = nacc/nits

    return store, acc

def optimal_policy_cov(
        x,
        theta0=0.0,
        theta1=0.0,
        theta2=0.0,
        theta3=1.0,
        theta4=2.5,
        theta5=0.0,
        theta6=2.5,
        theta7=0.0,
        theta8=1.0,
        theta9=2.5,
        theta10=0.0,
        theta11=2.5,
        theta12=0.0):
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

def nearestPD(A):
    """
    Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """
    Returns true when input is positive-definite, via Cholesky
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def make_nonsingular(matrix, epsilon=1e-6):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Check for near-zero eigenvalues
    near_zero_indices = np.abs(eigenvalues) < epsilon

    while isSingular(matrix):
        # Adjust near-zero eigenvalues by adding a small offset
        eigenvalues[near_zero_indices] += epsilon
        # Reconstruct the new matrix
        matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))

    return matrix

def isSingular(matrix):
    if np.linalg.det(matrix) <= 0:
        return True
    else:
        return False

def gauss_hermite_integral(func, n):
    xi, w = roots_hermite(n)
    integral = np.sum(w * func(xi))

    return integral

def exact_cumulative_reward(theta, q_func):
    npt = 20 # number of cubature nodes

    return (2 * np.pi)**(-0.5) * gauss_hermite_integral(partial(q_func, theta), npt) # cubature

def trunc_moment(L, U, x, m, s):
    l = (L - m) / s
    u = (U - m) / s
    Nl = norm.pdf(l, 0, 1)
    Nu = norm.pdf(u, 0, 1)
    Cl = norm.cdf(l, 0, 1)
    Cu = norm.cdf(u, 0, 1)

    if not (L == float('-inf') or U == float('inf')):
        out = (m - x)**2 * (Cu - Cl) + 2 * (m - x) * s * (Nl - Nu) \
              + s**2 * (Cu - Cl - (u * Nu - l * Nl))
    elif U == float('inf'):
        out = (m - x)**2 * (1 - Cl) + 2 * (m - x) * s * Nl \
              + s**2 * (1 - Cl + l * Nl)
    elif L == float('-inf'):
        out = (m - x)**2 * Cu - 2 * (m - x) * s * Nu \
              + s**2 * (Cu - u * Nu)

    return out

def Q(x, a):
    term1 = trunc_moment(-abs(x), abs(x), x, x, a)
    term2 = (norm.pdf(x, 0, np.sqrt(1 + a**2)) / norm.pdf(x, 0, 1)) \
            * (trunc_moment(-np.inf, -abs(x), x, x / (1 + a**2), a / np.sqrt(1 + a**2)) \
            + trunc_moment(abs(x), np.inf, x, x / (1 + a**2), a / np.sqrt(1 + a**2)))

    return term1 + term2

def omega(theta_curr, theta_prop, policy_func, add_noise_policy_func, log_p):
    """
    Importance weights for epsilon-greedy
    """
    sigma_curr = policy_func(theta_curr)
    sigma_prop = policy_func(theta_prop)
    noise_sigma_curr = add_noise_policy_func(theta_curr)
    noise_sigma_prop = add_noise_policy_func(theta_prop)

    # Log probability under policy_cov_func
    log_p_prop = log_p(theta_prop)
    log_p_curr = log_p(theta_curr)
    log_q_prop = norm.logpdf(theta_prop, loc=theta_curr, scale=sigma_curr)
    log_q_curr = norm.logpdf(theta_curr, loc=theta_prop, scale=sigma_prop)

    log_alpha = log_p_prop\
                - log_p_curr\
                + log_q_curr\
                - log_q_prop

    log_alpha = min(0.0, log_alpha)
    # print(alpha)
    prob_s = norm.logpdf(theta_curr, theta_prop, sigma_prop) + log_alpha

    # Log probability under noise_policy_cov_func
    noise_log_p_prop = log_p(theta_prop)
    noise_log_p_curr = log_p(theta_curr)
    noise_log_q_prop = norm.logpdf(theta_prop, loc=theta_curr, scale=noise_sigma_curr)
    noise_log_q_curr = norm.logpdf(theta_curr, loc=theta_prop, scale=noise_sigma_prop)
    noise_log_alpha = noise_log_p_prop\
                - noise_log_p_curr\
                + noise_log_q_curr\
                - noise_log_q_prop
    noise_log_alpha = min(0.0, noise_log_alpha)
    prob_s_pert = norm.logpdf(theta_curr, theta_prop, noise_sigma_prop) + noise_log_alpha

    # Importance weight

    print("prob_s", prob_s)
    print("prob_s_pert", prob_s_pert)

    print(np.exp(prob_s - prob_s_pert))

    return np.exp(prob_s - prob_s_pert)

def flat(nested_list):
    """
    Expand nested list
    """
    res = []
    for i in nested_list:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res

def first_nan_position(arr):
    nan_positions = np.isnan(arr)
    if np.any(nan_positions):
        return np.argmax(nan_positions)
    else:
        return None
