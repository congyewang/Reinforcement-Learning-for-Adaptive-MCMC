import jax.numpy as jnp
from jax import jit

@jit
def rbf_kernel(x, y, bandwidth=1.0):
    # Compute pairwise distances
    distances = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-distances / (2 * bandwidth * bandwidth))

def compute_mmd_final_efficient_jax(x, y, kernel='rbf', batch_size=10_000):
    """
    Final memory-efficient computation of MMD using JAX.
    
    Parameters:
        - x: First set of samples. Shape: (n, d)
        - y: Second set of samples. Shape: (m, d)
        - kernel: Type of kernel to be used. Currently supports 'rbf'.
        - batch_size: Size of batch to compute kernel values.
        
    Returns:
        - mmd: The MMD value between x and y.
    """
    n, _ = x.shape
    m = y.shape[0]

    def rbf_kernel(x, y, bandwidth=1.0):
        # Compute pairwise distances
        distances = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-distances / (2 * bandwidth * bandwidth))

    if kernel == 'rbf':
        kernel_func = rbf_kernel
    else:
        raise ValueError("Unsupported kernel type.")

    xx_sum, yy_sum, xy_sum = 0.0, 0.0, 0.0
    count_xx, count_yy, count_xy = 0.0, 0.0, 0.0  # Convert to float

    for i in range(0, n, batch_size):
        x_batch = x[i:min(i+batch_size, n)]
        
        xx_kernel_batch = kernel_func(x_batch, x_batch)
        xx_sum += jnp.sum(xx_kernel_batch)
        count_xx += float(x_batch.shape[0] * x_batch.shape[0])
        
        for j in range(0, m, batch_size):
            y_batch = y[j:min(j+batch_size, m)]
            
            xy_kernel_batch = kernel_func(x_batch, y_batch)
            xy_sum += jnp.sum(xy_kernel_batch)
            count_xy += float(x_batch.shape[0] * y_batch.shape[0])
            
            if i == j:
                yy_kernel_batch = kernel_func(y_batch, y_batch)
                yy_sum += jnp.sum(yy_kernel_batch)
                count_yy += float(y_batch.shape[0] * y_batch.shape[0])

    xx_kernel = xx_sum / count_xx
    yy_kernel = yy_sum / count_yy
    xy_kernel = xy_sum / count_xy

    # Compute MMD
    mmd = xx_kernel + yy_kernel - 2 * xy_kernel

    return mmd
