import numpy as np
import jax.numpy as jnp


class Distribution:
    @staticmethod
    def gaussian1D(x, mu=0.0, sigma=1.0):
        return -jnp.log(sigma*jnp.sqrt(2*jnp.pi)) - (x - mu)**2 / (2 * sigma**2)

    @staticmethod
    def gaussian2D(x, mu=0.0, sigma2=1.0):
        return -((x - mu)@(x - mu)) / (2 * sigma2)

    @staticmethod
    def banana(x, mu=100.0, sigma2=1.0):
        x0, x1 = x
        t1 = -x0**2 / (2 * sigma2)
        t2 = -((x1 - (x0**2 + mu))**2) / (2 * sigma2)
        return t1 + t2

    @staticmethod
    def gaussian_mixture(x, mu1=0.0, sigma21=1.0, mu2=0.0, sigma22=1.0, w1=0.5):
        return jnp.log(w1 * jnp.exp(Distribution.gaussian(x, mu1, sigma21)) + \
            (1 - w1) * jnp.exp(Distribution.gaussian(x, mu2, sigma22)))
