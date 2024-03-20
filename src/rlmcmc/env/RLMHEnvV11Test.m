classdef RLMHEnvV11Test < RLMHEnvBase
    methods
        function this = RLMHEnvV11Test(log_target_pdf, sample_dim)
            this = this@RLMHEnvBase(log_target_pdf, sample_dim);

            % Action specification
            this.ActionInfo = rlNumericSpec([bitshift(sample_dim, 1) + 1, 1]);
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

            % Calculate Log Target Density
            LogTargetCurrent = this.log_target_pdf(current_sample);
            LogTargetProposed = this.log_target_pdf(proposed_sample);

            % Calculate Log Proposal Density
            LogProposalCurrent = this.log_proposal_pdf(current_sample, proposed_mean, covariance);
            LogProposalProposed = this.log_proposal_pdf(proposed_sample, current_mean, covariance);

            % Calculate Log Acceptance Rate
            LogAlphaTemp = LogTargetProposed ...
                - LogTargetCurrent ...
                + LogProposalCurrent ...
                - LogProposalProposed;
            if isnan(LogAlphaTemp)
                log_alpha = -inf;
                disp([current_mean,proposed_mean,covariance])
                disp([LogTargetProposed,LogTargetCurrent,LogProposalCurrent,LogProposalProposed])
            else
                log_alpha = min( ...
                    0.0, ...
                    LogTargetProposed ...
                    - LogTargetCurrent ...
                    + LogProposalCurrent ...
                    - LogProposalProposed ...
                    );
            end

            % Accept or Reject
            if log(rand()) < log_alpha
                accepted_status = true;
                accepted_sample = proposed_sample;
                accepted_mean = proposed_mean;
                accepted_covariance = covariance;
            else
                accepted_status = false;
                accepted_sample = current_sample;
                accepted_mean = current_mean;
                accepted_covariance = covariance;
            end

            % Update Observation
            next_proposed_sample = mvnrnd(accepted_mean, accepted_covariance);
            observation = [accepted_sample; next_proposed_sample'];
            this.state = observation;

            % Store
            this.store_observation{end+1} = observation;

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