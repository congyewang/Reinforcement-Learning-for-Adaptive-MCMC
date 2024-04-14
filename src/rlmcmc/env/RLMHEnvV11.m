classdef RLMHEnvV11 < RLMHEnvBase
    methods
        function this = RLMHEnvV11(log_target_pdf, initial_sample, initial_covariance)
            this = this@RLMHEnvBase(log_target_pdf, initial_sample, initial_covariance);
            this.covariance = initial_covariance;

            % Action specification
            this.ActionInfo = rlNumericSpec([bitshift(this.sample_dim, 1) + 1, 1]);
            this.ActionInfo.Name = 'Act';
            this.ActionInfo.Description = 'a_{t} = [phi_{t}; phi^{*}_{t+1}; std]';
        end

    end

    methods
        function [observation, reward, is_done, info] = step(this, action)
            % Extract Current Sample
            current_sample = this.state(1:this.sample_dim);
            proposed_sample = this.state(this.sample_dim+1:end);

            % Extract Mean Drift and Covariance Magnification
            current_mean_drift = action(1:this.sample_dim);
            proposed_mean_drift = action(this.sample_dim+1:end-1);
            covariance_magnification = action(end);

            % Restore the Mean
            current_mean = current_sample + current_mean_drift;
            proposed_mean = proposed_sample + proposed_mean_drift;

            % Restore the Covariance
            covariance = (0.01 + covariance_magnification^2) * eye(this.sample_dim);

            % Accept or Reject
            [~, accepted_sample, accepted_mean, accepted_covariance, log_alpha] = this.accepted_process( ...
                current_sample, ...
                proposed_sample, ...
                current_mean, ...
                proposed_mean, ...
                covariance, ...
                covariance ...
                );

            % Update Observation
            next_proposed_sample = mvnrnd(accepted_mean, accepted_covariance);
            observation = [accepted_sample; next_proposed_sample'];
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

    end

end