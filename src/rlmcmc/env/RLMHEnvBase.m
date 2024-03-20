classdef (Abstract) RLMHEnvBase < rl.env.MATLABEnvironment
    properties
        sample_dim = nan; % sample dimension
        steps = 1; % iteration time
        state = []; % state at this time, s_{t}
        log_target_pdf_pointer = nan; % log target probability density function

        % Store
        store_observation = {};
        store_action = {};
        store_log_accetance_rate = {};
        store_accepted_status = {};
        store_reward = {};

        store_current_sample = {};
        store_proposed_sample = {};

        store_current_covariance = {};
        store_proposed_covariance = {};

        store_log_target_proposed = {};
        store_log_target_current = {};
        store_log_proposal_proposed = {};
        store_log_proposal_current = {};

        store_accepted_mean = {};
        store_accepted_sample = {};
        store_accepted_covariance = {};
    end

    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false
    end

    methods
        function this = RLMHEnvBase(log_target_pdf, sample_dim)
            assert(sample_dim>0, "The sample dimension must be greater than 0.")

            % Observation specification
            observation_info = rlNumericSpec([bitshift(sample_dim, 1), 1]);
            observation_info.Name = 'Obs';
            observation_info.Description = 's_{t} = [x_{t}; x^{*}_{t+1}]';

            % Action specification
            action_info = rlNumericSpec([bitshift(sample_dim, 1), 1]);
            action_info.Name = 'Act';
            action_info.Description = 'a_{t} = [mean_{t}; mean^{*}_{t+1}]';

            % The following line implements built-in functions of rl.env.VariantEnv
            this = this@rl.env.MATLABEnvironment(observation_info, action_info);

            % Initialize Sample Dimension and State
            this.sample_dim = sample_dim; % sample dimension
            initial_sample = mvnrnd(0, 1, sample_dim);
            initial_next_proposed_sample = mvnrnd(0, 1, sample_dim);
            this.state = [initial_sample;initial_next_proposed_sample];

            % Pass in the Log Target PDF pointer
            this.log_target_pdf_pointer = log_target_pdf;
        end
    end

    methods
        function res = log_target_pdf(this, x)
            res = this.log_target_pdf_pointer(x);
        end

        function res = log_proposal_pdf(this, x, mean, var)
            res = logmvnpdf(x', mean', var);
        end

        function reward = reward_function(this, current_sample, proposed_sample, log_alpha, log_mode)
            if nargin < 5
                log_mode = true;
            end

            if log_mode
                reward = 2 * log(norm(current_sample - proposed_sample, 2)) + log_alpha;
            else
                reward = norm(current_sample - proposed_sample, 2)^2 * exp(log_alpha);
            end
        end

        function [accepted_status, accepted_sample, accepted_mean, accepted_covariance, log_alpha] = accepted_process( ...
                this, ...
                current_sample, ...
                proposed_sample, ...
                current_mean, ...
                proposed_mean, ...
                current_covariance, ...
                proposed_covariance ...
                )

            %% Calculate Log Target Density
            log_target_current = this.log_target_pdf(current_sample);
            log_target_proposed = this.log_target_pdf(proposed_sample);

            %% Calculate Log Proposal Densitys
            log_proposal_current = this.log_proposal_pdf(current_sample, proposed_mean, proposed_covariance);
            log_proposal_proposed = this.log_proposal_pdf(proposed_sample, current_mean, current_covariance);

            %% Calculate Log Acceptance Rate
            log_alpha = min( ...
                0.0, ...
                log_target_proposed ...
                - log_target_current ...
                + log_proposal_current ...
                - log_proposal_proposed ...
                );

            %% Accept or Reject
            if log(rand()) < log_alpha
                accepted_status = true;
                accepted_sample = proposed_sample;
                accepted_mean = proposed_mean;
                accepted_covariance = proposed_covariance;
            else
                accepted_status = false;
                accepted_sample = current_sample;
                accepted_mean = current_mean;
                accepted_covariance = current_covariance;
            end

            %% Store
            % Store Covariance
            this.store_current_covariance{end+1} = current_covariance;
            this.store_proposed_covariance{end+1} = proposed_covariance;

            % Store Log Densities
            this.store_log_target_current{end+1} = log_target_current;
            this.store_log_target_proposed{end+1} = log_target_proposed;

            this.store_log_proposal_current{end+1} = log_proposal_current;
            this.store_log_proposal_proposed{end+1} = log_proposal_proposed;

            % Store Acceptance
            this.store_accepted_status{end+1} = accepted_status;
            this.store_log_accetance_rate{end+1} = log_alpha;

            this.store_accepted_sample{end+1} = accepted_sample;
            this.store_accepted_mean{end+1} = accepted_mean;
            this.store_accepted_covariance{end+1} = accepted_covariance;
        end

        function initial_observation = reset(this)
            %{
            Reset environment to initial state and return initial observation
            %}
            initial_observation = this.state;
        end
    end

    methods (Abstract)
        [observation, reward, is_done, info] = step(this, action)
    end

end
