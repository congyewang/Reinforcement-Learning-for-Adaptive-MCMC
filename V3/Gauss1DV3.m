classdef Gauss1DV3 < rl.env.MATLABEnvironment
    properties
        sigma = 1; % std of the proposal pdf
        % TotalTimeSteps = 10000; % total iteration
        Reward = [];
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
        function this = Gauss1DV3()
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

        function [logp] = logmvnpdf(this, x,mu,Sigma)
            % outputs log likelihood array for observations x  where x_n ~ N(mu,Sigma)
            % x is NxD, mu is 1xD, Sigma is DxD
            [N,D] = size(x);
            const = -0.5 * D * log(2*pi);
            xc = bsxfun(@minus,x,mu);
            term1 = -0.5 * sum((xc / Sigma) .* xc, 2); % N x 1
            term2 = const - 0.5 * this.logdet(Sigma);    % scalar
            logp = term1' + term2;
        end

        function y = logdet(this, A)
            U = chol(A);
            y = 2*sum(log(diag(U)));
        end

        function [lse,sm] = logsumexp(this, x)
            %LOGSUMEXP  Log-sum-exp function.
            %    lse = LOGSUMEXP(x) returns the log-sum-exp function evaluated at
            %    the vector x, defined by lse = log(sum(exp(x)).
            %    [lse,sm] = LOGSUMEXP(x) also returns the softmax function evaluated
            %    at x, defined by sm = exp(x)/sum(exp(x)).
            %    The functions are computed in a way that avoids overflow and
            %    optimizes numerical stability.

            %    Reference:
            %    P. Blanchard, D. J. Higham, and N. J. Higham.
            %    Accurately computing the log-sum-exp and softmax functions.
            %    IMA J. Numer. Anal., Advance access, 2020.

            if ~isvector(x), error('Input x must be a vector.'), end

            n = length(x);
            s = 0; e = zeros(n,1);
            [xmax,k] = max(x); a = xmax;
            s = 0;
            for i = 1:n
                e(i) = exp(x(i)-xmax);
                if i ~= k
                    s = s + e(i);
                end
            end
            lse = a + log1p(s);
            if nargout > 1
                sm = e/(1+s);
            end
        end

        function [res] = logTargetPdf(this, x)
            weight1 = 0.5;
            weight2 = 0.5;

            % Calculate logpdf
            log_pdf1 = log(normpdf(x, -3, 1));
            log_pdf2 = log(normpdf(x, 3, 1));

            % logpdf Plus Weights
            weighted_log_pdf1 = log(weight1) + log_pdf1;
            weighted_log_pdf2 = log(weight2) + log_pdf2;

            res = this.logsumexp([weighted_log_pdf1, weighted_log_pdf2]);
            % res = this.logmvnpdf(x, 0, 1);

        end

        function [res] = logProposalPdf(this, x, mu, sigma)
            res = this.logmvnpdf(x, mu, sigma);
        end

        function [reward] = getReward(this, current_sample, proposed_sample, log_alpha)
            reward = (1/2)*log(norm(current_sample - proposed_sample)) + log_alpha;
        end

        function [Observation, Reward, IsDone, Info] = step(this, Action)
            Info = [];

            % Unpack action
            phi_x_n = Action(1);
            phi_x_n_plus_1 = Action(2);

            % Covariance Restore
            cov_n = phi_x_n^2 * eye(1);
            cov_n_plus_1 = phi_x_n_plus_1^2 * eye(1);

            % Unpack state
            x_n = this.State(1);
            x_n_plus_1 = this.State(2);

            % Calculate Log Target Density
            LogTargetCurrent = this.logTargetPdf(x_n);
            LogTargetProposed = this.logTargetPdf(x_n_plus_1);

            % Calculate Log Proposal Density
            LogProposalCurrent = this.logProposalPdf(x_n, x_n_plus_1, cov_n_plus_1);
            LogProposalProposed = this.logProposalPdf(x_n_plus_1, x_n, cov_n);

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
                AcceptedSample = x_n_plus_1;
                AcceptedCov = cov_n_plus_1;
            else
                AcceptedStatus = false;
                AcceptedSample = x_n;
                AcceptedCov = cov_n;
            end

            % Update Observation
            NextProposedSample = normrnd(AcceptedSample, sqrt(AcceptedCov));
            Observation = [AcceptedSample; NextProposedSample];
            this.State = Observation;

            % Calculate Reward
            Reward = this.getReward(x_n, x_n_plus_1, LogAlpha);

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
