classdef GaussMix1D < rl.env.MATLABEnvironment
    properties
        % Parameter
        sigma = 1;
        epsilon = 0.01;
        nsamples = 1;
        MaxSteps = 100;
        Reward = 0;
        Ts = 0; % iteration time
        State = 0; % state at this time, s_{t}
        OldState = 0; % state at previous state, s_{t-1}

        % Store
        StoreState = {};
        StoreAction = {};
    end

    methods
        function this = GaussMix1D()
            % Observation specification
            ObservationInfo = rlNumericSpec([1 1]);
            ObservationInfo.Name = 'Obs';
            ObservationInfo.Description = 'Description of the observation';

            % Action specification
            ActionInfo = rlNumericSpec([1 1],'LowerLimit',0,'UpperLimit',inf);
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

            pdf = @(x) 0.3*normpdf(x, 4, 1) + 0.7*normpdf(x, -5, 3);
            proprnd = @(x) x + this.sigma * randn(1, 1);
            xt = mhsample(this.State,this.nsamples,'pdf',pdf,'proprnd',proprnd,'symmetric',this.epsilon);

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
            this.State = 0; % initialize s_{t}
            this.OldState = 0; % initialize s_{t-1}
            obs = this.State;
        end
    end
end
