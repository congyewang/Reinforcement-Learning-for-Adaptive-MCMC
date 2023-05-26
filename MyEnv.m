classdef MyEnv < rl.env.MATLABEnvironment
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
    end

    methods
        function this = MyEnv()
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

            pdf = @(x) normpdf(x, 4, 1);
            proprnd = @(x) x + this.sigma * randn(1, 1);
            xt = mhsample(this.State,this.nsamples,'pdf',pdf,'proprnd',proprnd,'symmetric',this.epsilon);

            % Update Obs
            NextObs = xt;
            this.OldState = this.State; % Save s_{t-1}
            this.State = xt; % Update xt in this state

            % Calculate Reward
            Reward = pdist2(this.State - this.OldState);

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
