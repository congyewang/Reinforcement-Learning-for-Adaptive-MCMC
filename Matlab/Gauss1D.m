classdef Gauss1D < rl.env.MATLABEnvironment
    properties
        % Parameter
        sigma = 1.0;
        MaxSteps = 1000;
        Reward = 0.0;
        Ts = 0; % iteration time
        State = 0.0; % state at this time, s_{t}
        OldState = 0.0; % state at previous state, s_{t-1}

        % Store
        StoreState = {0.0};
        StoreAction = {1.0};
    end

    methods
        function this = Gauss1D()
            % Observation specification
            ObservationInfo = rlNumericSpec([1 1]);
            ObservationInfo.Name = 'Obs';
            ObservationInfo.Description = 'Description of the observation';

            % Action specification
            ActionInfo = rlNumericSpec([1 1],'LowerLimit',-inf,'UpperLimit',inf);
            ActionInfo.Name = 'Act';

            % The following line implements built-in functions of rl.env.VariantEnv
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);

            % Initial State
            this.reset();
        end

        function [NextObs,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];

            % Update Action
            this.sigma = Action;
            
            disp(Action);

            logpdf = @(x) log(normpdf(x, 4, 1));
            % xt = mhsample(this.State,this.nsamples,'pdf',pdf,'proprnd',proprnd,'symmetric',this.epsilon);
            sigma2_curr = (this.StoreAction{end})^2;
            sigma2_prop = Action^2;
            % xt = asymmetric1D(sigma2_curr, this.sigma^2, this.State, logpdf);
            % proprnd = @(x) x + abs(this.sigma) * randn(1, 1);
            xt = rwm(sigma2_curr, sigma2_prop, this.State, logpdf, 1);

            % Update Obs
            NextObs = xt;
            this.OldState = this.State; % Save s_{t-1}
            this.State = xt; % Update xt in this state

            % Print State
            % fprintf('State: %.4f\tAction: %.4f\n', this.State, this.sigma);
            this.StoreState{end+1} = xt;
            this.StoreAction{end+1} = this.sigma;

            % Calculate Reward
            Reward = norm(this.State - this.OldState, 2)^2;

            % Update Iteration Time
            this.Ts = this.Ts + 1;

            % Check for Completion
            IsDone = this.Ts >= this.MaxSteps;

            if IsDone
                this.reset();
            end
        end

        function obs = reset(this)
            this.Ts = 0;
            [~, maxIndex] = max(cell2mat(this.StoreState));
            this.State = this.StoreState{maxIndex}; % initialize s_{t}
            this.OldState = this.StoreState{maxIndex}; % initialize s_{t-1}
            obs = this.State;
        end
    end
end
