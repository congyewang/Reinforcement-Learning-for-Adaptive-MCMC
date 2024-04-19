classdef RLMHEnv < RLMHEnvBase
    properties
        covariance = nan;
    end

    methods
        function this = RLMHEnv(log_target_pdf, initial_sample, initial_covariance)
            this = this@RLMHEnvBase(log_target_pdf, initial_sample, initial_covariance);
            this.covariance = initial_covariance;

            % Action specification
            this.ActionInfo = rlNumericSpec([bitshift(this.sample_dim, 1), 1]);
            this.ActionInfo.Name = 'Act';
            this.ActionInfo.Description = 'a_{t} = [phi_{t}; phi^{*}_{t+1}]';
        end
    end

    methods
        function res = log_proposal_pdf(this, x, mu, var)
            res = logmvlpdf(x', mu', var);
            % res = logmvnpdf(x', mu', var);
        end
    end

    methods
        function [observation, reward, is_done, info] = step(this, action)
            % Extract Current Sample
            current_sample = this.state(1:this.sample_dim);
            proposed_sample = this.state(this.sample_dim+1:end);

            % Extract Mean Drift and Covariance Magnification
            current_mean = action(1:this.sample_dim);
            proposed_mean = action(this.sample_dim+1:end);

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
            next_proposed_sample = mvlaprnd(this.sample_dim, accepted_mean, accepted_covariance);
            % next_proposed_sample = mvnrnd(accepted_mean, accepted_covariance)';
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

    end

end
