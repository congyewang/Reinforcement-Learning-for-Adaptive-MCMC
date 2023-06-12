import torch


def rwm_env(sigma, theta_start, log_pi, nits=1):
    """
    Random Walk Metropolis-Hastings Algorithm
    """
    d = theta_start.shape[0]
    store = torch.zeros((nits+1, d))
    theta_curr = theta_start
    log_pi_curr = log_pi(theta_curr)
    store[0,:] = theta_curr
    for i in range(nits):
        psi = theta_curr + sigma * torch.distributions.normal.Normal(0, 1).sample()
        log_pi_prop = log_pi(psi)
        log_alpha = log_pi_prop - log_pi_curr
        if torch.log(torch.rand(1)) < log_alpha:
            theta_curr = psi
            log_pi_curr = log_pi_prop

        store[i+1,:] = theta_curr

    return store[1:]


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    sigma = 1
    theta_start = torch.tensor([0.])
    log_pi = lambda x: -(x-5)**2/2
    nits = 10000
    theta = rwm_env(sigma, theta_start, log_pi, nits)

    plt.plot(theta)
    plt.show()
