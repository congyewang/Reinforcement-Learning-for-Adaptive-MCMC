classdef Gauss1DV8 < rl.env.MATLABEnvironment
    properties
        % Parameter
        sigma = 1.0; % std of the proposal pdf
        % TotalTimeSteps = 10000; % total iteration
        Reward = 0.0;
        Ts = 1; % iteration time

        State = [0; randn]; % state at this time, s_{t}

        % Store
        StoreState = {};
        StoreAction = {};
        StoreAcceptedStatus = {};
        StoreReward = {};
    end

    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false
    end

    methods
        function this = Gauss1DV8()
            % Observation specification
            ObservationInfo = rlNumericSpec([2, 1]);
            ObservationInfo.Name = 'Obs';
            ObservationInfo.Description = 's_{t} = [x_{t}; x^{*}_{t+1}]';

            % Action specification
            ActionInfo = rlNumericSpec([2, 1]);
            ActionInfo.Name = 'Act';
            ActionInfo.Description = 'a_{t} = [mean_{t}; mean^{*}_{t+1}]';

            % The following line implements built-in functions of rl.env.VariantEnv
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);

            % % Initial State
            % this.reset();
        end

        function [res] = logTargetPdf(this, x, mu, sigma)
            if nargin < 3
                mu = 0;
            end
            if nargin < 4
                sigma = 1;
            end

            res = log(normpdf(x, mu, sigma));
        end

        function [res] = logProposalPdf(this, x, mu, sigma)
            res = log(normpdf(x, mu, sigma));
        end

        function [dist] = getDistance(this, current_sample, proposed_sample)
            dist = norm(current_sample - proposed_sample, 2);
        end

        function [reward] = getReward(this, current_sample, proposed_sample, log_alpha)
            reward = this.getDistance(current_sample, proposed_sample)^2 * exp(log_alpha);
        end

        function [Observation, Reward, IsDone, Info] = step(this, Action)
            Info = [];

            % Unpack action
            CurrentMean = Action(1);
            ProposedMean = Action(2);

            % Unpack state
            CurrentSample = this.State(1);
            ProposedSample = this.State(2);

            % Calculate Log Target Density
            LogTargetCurrent = this.logTargetPdf(CurrentSample);
            LogTargetProposed = this.logTargetPdf(ProposedSample);

            % Calculate Log Proposal Density
            LogProposalCurrent = this.logProposalPdf(CurrentSample, ProposedSample, this.sigma);
            LogProposalProposed = this.logProposalPdf(ProposedSample, CurrentSample, this.sigma);

            % Calculate Log Acceptance Rate
            LogAlphaTemp = LogTargetProposed ...
                - LogTargetCurrent ...
                + LogProposalCurrent ...
                - LogProposalProposed;
            if isnan(LogAlphaTemp)
                LogAlpha = -inf;
            else
                LogAlpha = min( ...
                    0.0, ...
                    LogTargetProposed ...
                    - LogTargetCurrent ...
                    + LogProposalCurrent ...
                    - LogProposalProposed ...
                    );
            end

            % Accept or Reject
            if log(rand()) < LogAlpha
                AcceptedStatus = true;
                AcceptedSample = ProposedSample;
                AcceptedMean = ProposedMean;
            else
                AcceptedStatus = false;
                AcceptedSample = CurrentSample;
                AcceptedMean = CurrentMean;
            end

            % Update Observation
            NextProposedSample = normrnd(AcceptedMean, this.sigma);
            Observation = [AcceptedSample; NextProposedSample];
            this.State = Observation;

            % Calculate Reward
            if isnan(LogAlphaTemp)
                Reward = 0.0;
            else
                Reward = this.getReward(CurrentSample, ProposedSample, LogAlpha);
            end

            % Store
            this.StoreState{end+1} = Observation;
            this.StoreAction{end+1} = Action;
            this.StoreAcceptedStatus{end+1} = AcceptedStatus;
            this.StoreReward{end+1} = Reward;

            % Check terminal condition
            % IsDone = this.Step >= this.TotalTimeSteps;
            % this.IsDone = IsDone;
            IsDone = false;
            this.IsDone = IsDone;

            % Update Step
            this.Ts = this.Ts + 1;
        end

        % Reset environment to initial state and return initial observation
        function InitialObservation = reset(this)
            InitialObservation = this.State;
        end
    end
end
