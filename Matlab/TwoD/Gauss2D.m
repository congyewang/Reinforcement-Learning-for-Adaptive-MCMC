classdef Gauss2D < rl.env.MATLABEnvironment
    properties
        % Parameter
        sigma2 = eye(2);
        MaxSteps = 500;
        Reward = 0;
        Ts = 0; % iteration time
        State = zeros(2,1); % state at this time, s_{t}
        OldState = zeros(2,1); % state at previous state, s_{t-1}

        % Store
        StoreState = {zeros(2,1)};
        StoreAction = {eye(2)};
        StoreAcceptedStatus = {1};
    end

    methods
        function this = Gauss2D()
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
            logpdf = @(x) log(mvnpdf(x, [0,0], eye(2)));
            sigma2_curr = this.StoreAction{end};
            [xtT,accepted_status] = rwm(sigma2_curr, this.sigma2, this.State.', logpdf, 1);
            xt = xtT.';

            this.StoreAcceptedStatus{end+1} = accepted_status;
            % Update Obs
            NextObs = xt;
            this.OldState = this.State; % Save s_{t-1}
            this.State = xt; % Update xt in this state

            % Print State
            % fprintf('State: %.4f\tAction: %.4f\n', this.State, this.sigma);
            this.StoreState{end+1} = xt;
            this.StoreAction{end+1} = this.sigma2;
            disp(this.sigma2);

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
