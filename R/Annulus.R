library(rstan)
library(ggplot2)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


data <- list(N = 2, r0 = 10, sigma = 1)

fit <- stan(file = './models/Annulus.stan',
            data = data,
            chains = 1,
            iter = 100000)
samples_matrix <- extract(fit, permuted = FALSE)
samples_df <- as.data.frame(samples_matrix)
names(samples_df) <- c("x1", "x2", "lp")
p <- ggplot(samples_df, aes(x=x1, y=x2)) +
  geom_point()
p
