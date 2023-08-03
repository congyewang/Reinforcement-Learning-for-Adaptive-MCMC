clear all
clf
rng(1)
use_latex

%% policy
sigma = @(x,theta) theta(1)^2 + theta(2)^2 * abs(x);
dsigma_theta = @(x,theta) [2*theta(1), 2*theta(2) * abs(x)]';

%% initialise
x0 = 100;
X(1) = x0;
theta(:,1) = [0.01;0.01];
gamma = 0.9; % discount factor

I = 10;
epi_length = 20;
for i = 1:I % episodes

   %% auxiliary chains
   J = 50; % number of parallel "auxiliary" chains
   aux_length = 20;
   rew_length = 5; % truncate the cumulative reward
   x0_aux = X((i-1)*epi_length+1); % current state of the "main" chain
   for j = 1:J 
        X_aux(:,j,i) = MCMC(x0_aux, ...
                            @(x) sigma(x,theta(:,i)), ...
                            aux_length); % states of the jth "auxiliary" chain
   end
   
   %% approximate the Q function (for the current policy)
   % collect together all of the data for training of Q
   for j = 1:J
       for k = 1:aux_length-rew_length
           x(j,k) = X_aux(k,j,i);
           a(j,k) = sigma(x(j,k),theta(:,i));
           y(j,k) = cum_r(X_aux(k:(k+rew_length),j,i),gamma); % cumulative reward of jth "auxiliary" chain
       end
   end
   % fit Q
   Q_w = @(x,a,w) w(1) + w(2) * abs(a * x);
   dQ_a = @(x,a,w) w(2) * sign(a) * abs(x);
   A = [ones(J*(aux_length-rew_length),1) , a(:) .* x(:)];
   w = lsqr(A,y(:)); 
   
   %% plotting
   figure(1)
   set(gcf,'color','w')
   set(gcf,'Position',[100 100 600 900])
   plot_interval = [min(x(:)),max(x(:)),min(a(:)),max(a(:))];
   if ismember(i,[1,2,I]) % plot just for certain episodes, as it's quite time-consuming
       if ismember(i,[1,2])
           subplot(3,2,2*(i-1)+1)
       else
           subplot(3,2,5)
       end
       fcontour(@(x,a) log(Q(x,a,@(y) sigma(y,theta(:,i)),rew_length,gamma)),plot_interval);
       xlabel('$x$')
       ylabel('$\varphi$')
       legend({'true $\log(Q)$'})
       title(['episode ',num2str(i,'%u')])
       if ismember(i,[1,2])
           subplot(3,2,2*(i-1)+2)
       else
           subplot(3,2,6)
       end
       fcontour(@(x,a) log(Q_w(x,a,w)),plot_interval)
       hold on
       plot(x(:),a(:),'rx')
       vline(x0_aux,'k-')
       hline(sigma(x0_aux,theta(:,i)),'k-')
       xlabel('$x$')
       ylabel('$\varphi$')
       legend({'approx $\log(Q)$','training data'})
       title(['episode ',num2str(i,'%u')])
       if i == I
           exportgraphics(gcf,'Q.pdf')
       end
   end
   
   %% policy gradient
   V = 0;
   for j = 1:J
       for k = 1:20
           V = V ...
                 + (1/J) * gamma^(k-1) * dsigma_theta(X_aux(k,j,i),theta(:,i)) ...
                         * dQ_a(X_aux(k,j,i),sigma(X_aux(k,j,i),theta(:,i)),w);
       end
   end
   
   %% update the policy
   alpha = 0.001; % learning rate (here fixed)
   theta(:,i+1) = theta(:,i) + alpha * V;
   
   %% main chain
   X((i-1)*epi_length+1:i*epi_length+1) = MCMC(X((i-1)*epi_length+1), ...
                                             @(x) sigma(x,theta(:,i+1)), ...
                                             epi_length+1);

end

%% plotting
figure(2)
set(gcf,'color','w')
set(gcf,'Position',[100 100 600 300])
subplot(1,2,1)
plot(X,'b-')
hold on
for i = 1:I
    vline((i-1)*epi_length,'k')
    for j = 1:J
        plot(((i-1)*epi_length+1):((i-1)*epi_length+aux_length),X_aux(:,j,i),'r:') 
    end
end
xlabel('iteration')
ylabel('$x$')
title('Chains')
legend({'Main','Aux'})
subplot(1,2,2)
plot(theta(1,:).^2,'b-')
hold on
plot(theta(2,:).^2,'r-')
legend({'$\theta_1^2$','$\theta_2^2$'})
xlabel('episode $i$')
title('Policy Parameters')
exportgraphics(gcf,'trace.pdf')







