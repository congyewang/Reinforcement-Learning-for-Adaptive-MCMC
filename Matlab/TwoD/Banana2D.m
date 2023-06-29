classdef Banana2D < rl.env.MATLABEnvironment
    properties
        % Parameter
        sigma2 = eye(2);
        nsamples = 1;
        MaxSteps = 100;
        Reward = 0;
        Ts = 0; % iteration time
        State = zeros(2,1); % state at this time, s_{t}
        OldState = zeros(2,1); % state at previous state, s_{t-1}

        % Store
        StoreState = {};
        StoreAction = {};
    end

    methods
        function this = Banana2D()
            % Observation specification
            ObservationInfo = rlNumericSpec([2 1]);
            ObservationInfo.Name = 'Obs';
            ObservationInfo.Description = 'Description of the observation';

            % Action specification
            ActionInfo = rlNumericSpec([3 1],'LowerLimit',-inf*ones(3,1),'UpperLimit',inf*ones(3,1));
            ActionInfo.Name = 'Act';

            % The following line implements built-in functions of rl.env.VariantEnv
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);

            % Initial State
            this.reset();
        end

        function [NextObs,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];

            % Update Action
            a = Action(1);
            b = Action(2);
            c = Action(3);
            this.sigma2 = [a, 0; b, c] * [a, b; 0, c];

            % Define the banana distribution
            % pdf = @(x) exp(-100*(x(2)-x(1)^2)^2-(1-x(1))^2);
            logpdf = @(x) -5*(x(2)-x(1)^2)^2-(-x(1))^2;
            % proprnd = @(x) mvnrnd(x, this.sigma2, 1);
            proprnd = @(x) mvnrnd(x, this.sigma2, 1);
            xt = mhsample(this.State.',this.nsamples,'logpdf',logpdf,'proprnd',proprnd,'symmetric',1);

            % Update Obs
            NextObs = xt.';
            this.OldState = this.State; % Save s_{t-1}
            this.State = xt.'; % Update xt in this state

            % Print State
            % fprintf('State: %.4f\tAction: %.4f\n', this.State, this.sigma);
            this.StoreState{end+1} = xt.';
            this.StoreAction{end+1} = this.sigma2;
            disp(this.sigma2);
    
            % Calculate Reward
            Reward = norm(this.State - this.OldState, 2)^2;

            % Update Iteration Time
            this.Ts = this.Ts + 1;

            % Check for Completion
            IsDone = false;
        end

        function obs = reset(this)
            this.Ts = 0;
            % this.State = zeros(2,1); % initialize s_{t}
            % this.OldState = zeros(2,1); % initialize s_{t-1}
            obs = this.State;
        end
    end
end
