library(rstan)
library(ggplot2)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


data <- list(
  uniform_areas = matrix(
    c(-4, -2, -4, -2,
      0, 2, -4, -2,
      -2, 0, -2, 0,
      2, 4, -2, 0,
      -4, -2, 0, 2,
      0, 2, 0, 2,
      -2, 0, 2, 4,
      2, 4, 2, 4),
    nrow = 8, ncol = 4, byrow = TRUE)
)

initial_values <- list(
  list(mu = c(-1, -1)),
  list(mu = c(-3, -3))
)

fit <- stan(file = './models/checkerboard.stan',
            data = data,
            chains = 2,
            iter = 1000,
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
