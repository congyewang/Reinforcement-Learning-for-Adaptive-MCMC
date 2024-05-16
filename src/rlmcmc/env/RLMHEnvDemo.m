classdef RLMHEnvDemo < RLMHEnvBase
    properties
        target_mean = [];
        covariance = [];
    end

    methods
        function this = RLMHEnvDemo(log_target_pdf, initial_sample, target_mean, initial_covariance)
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
        function [observation, reward, is_done, info] = step(this, action)
            % Unpack action
            current_phi = action(1:this.sample_dim);
            proposed_phi = action(this.sample_dim+1:end);

            % Unpack state
            current_sample = this.state(1:this.sample_dim);
            proposed_sample = this.state(this.sample_dim+1:end);

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
