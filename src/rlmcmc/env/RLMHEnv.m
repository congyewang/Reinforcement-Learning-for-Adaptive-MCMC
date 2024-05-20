classdef RLMHEnv < RLMHEnvBase
    properties
        target_mean = [];
        covariance = [];
    end

    methods
        function this = RLMHEnv(log_target_pdf, initial_sample, target_mean, initial_covariance)
            this = this@RLMHEnvBase(log_target_pdf, initial_sample, initial_covariance);

            % Action specification
            this.ActionInfo = rlNumericSpec([bitshift(this.sample_dim, 1), 1]);
            this.ActionInfo.Name = 'Act';
            this.ActionInfo.Description = 'a_{t} = [phi_{t}; phi^{*}_{t+1}]';

            % Target Mean and Covariance
            this.target_mean = target_mean;
            this.covariance = initial_covariance;
        end
    end

    methods
        function res = gamma_c(this, x, mu, Sigma, c)
            if nargin < 5
                c = 10;
            end
            Sig_half = sqrtm(Sigma);
            eta = norm(Sig_half \ (x - mu))^2 / c^2;

            if eta >= 0 && eta < 0.5
                res = 0;
            elseif eta >= 0.5 && eta < 1
                res = (1 + exp(-((4 * eta - 3) / (4 * eta^2 - 6 * eta + 2))))^(-1);
            else
                res = 1;
            end
        end
    end

    methods
        function [observation, reward, is_done, info] = step(this, action)
            % Unpack state
            current_sample = this.state(1:this.sample_dim);
            proposed_sample = this.state(this.sample_dim+1:end);

            % Unpack action
            current_psi = action(1:this.sample_dim);
            proposed_psi = action(this.sample_dim+1:end);

            current_phi = current_psi + this.gamma_c( ...
                current_sample, ...
                this.target_mean, ...
                this.covariance) * (current_sample - current_psi);
            proposed_phi = proposed_psi + this.gamma_c( ...
                proposed_sample, ...
                this.target_mean, ...
                this.covariance)*(proposed_sample - proposed_psi);

            % Proposal means
            proposal_covariance_root = sqrtm(this.covariance);
            current_mean = this.target_mean + proposal_covariance_root * current_phi;
            proposed_mean = this.target_mean + proposal_covariance_root * proposed_phi;

            % Accept or Reject
            [~, accepted_sample, accepted_mean, accepted_covariance, log_alpha] = this.accepted_process( ...
                current_sample, ...
                proposed_sample, ...
                current_mean, ...
                proposed_mean, ...
                this.covariance, ...
                this.covariance ...
                );

            % Update Observation
            next_proposed_sample = Laplace(accepted_mean, accepted_covariance);
            observation = [accepted_sample; next_proposed_sample];
            this.state = observation;

            % Store
            this.store_observation{end+1} = observation;
            this.store_action{end+1} = action;

            % Calculate Reward
            reward = this.reward_function(current_sample, proposed_sample, log_alpha);
            this.store_reward{end+1} = reward;

            % Update Steps
            this.steps = this.steps + 1;
            is_done = false;
            info = [];
        end

        % Reset environment
        function InitialObservation = reset(this)
            InitialObservation = this.state;
        end
    end
end
