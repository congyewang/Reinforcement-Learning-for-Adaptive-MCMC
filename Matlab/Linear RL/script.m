clear all
close all
use_latex

%% plot the true Q function
figure(1)
set(gcf,'color','w')
subplot(2,3,1)
fcontour(@Q,[-3,3,0.1,5],'LevelStep',0.1)
xlabel('$x$')
ylabel('$\varphi$')
title('True $Q(x,\varphi)$')

%% plot the true J function
subplot(2,3,2)
fplot(@J,[0,5])
xlabel('$\theta$')
ylabel('$J(\theta)$')
title('True $J(\theta)$')

%% verify the policy gradient is correct
syms x a
Q_da = gradient(Q(x,a),a); % symbolic gradient of Q function wrt action
Q_da = matlabFunction(Q_da,'Vars',[x,a]);
npt = 20;
J_dt = @(t) (2*pi)^(-1/2) * GaussHermite(@(x) Q_da(x,t),npt); % integrate wrt P
subplot(2,3,3)
fplot(J_dt,[0,5])
xlabel('$\theta$')
ylabel('$(d/d\theta) J(\theta)$')
title('Verify the Policy Gradient Theorem')

%% run our method
[x,t,w] = RL_MCMC();

%% plot the results
subplot(2,3,4)
plot(x)
xlabel('its')
ylabel('$x$')
subplot(2,3,5)
plot(t)
xlabel('its')
ylabel('$\theta$')
subplot(2,3,6)
plot(w)
xlabel('its')
ylabel('$w$')