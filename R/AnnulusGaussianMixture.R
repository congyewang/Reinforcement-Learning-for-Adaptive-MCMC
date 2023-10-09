library(rstan)
library(ggplot2)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


data <- list(
  N = 2,
  w1 = 0.3,
  w2 = 0.7,
  mean = c(0.0, 0.0),
  cov = matrix(c(1.0, 0.0, 0.0, 1.0), nrow=2, ncol=2),
  r0 = 20.0,
  sigma = 1.0
)

initial_values <- list(
  list(mu = c(0, 0)),
  list(mu = c(20, 20))
)

fit <- stan(file = './models/Annulus_Gaussian_Mixture.stan',
            data = data,
            chains = 2,
            iter = 100000,
            init = initial_values,
            warmup = 0
            )

df_total <- as.data.frame(fit)
names(df_total) <- c("x1", "x2", "lp")

samples_matrix <- extract(fit, permuted = FALSE)
df_samples <- as.data.frame(samples_matrix)
names(df_samples) <- c("x_c1_1", "x_c2_1", "x_c1_2", "x_c2_2", "c1_lp", "c2_lp")

p_total <- ggplot(df_total, aes(x=x1, y=x2)) +
  geom_point()
p_total

p_chain1 <- ggplot(df_samples, aes(x=x_c1_1, y=x_c1_2)) +
  geom_point()
p_chain1

p_chain2 <- ggplot(df_samples, aes(x=x_c2_1, y=x_c2_2)) +
  geom_point()
p_chain2
