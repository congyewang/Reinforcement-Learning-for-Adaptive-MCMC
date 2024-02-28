classdef Gauss1DV9 < rl.env.MATLABEnvironment
    properties (Constant, Access = protected)
        INF = 1.7977e+307; % 0.1 * inf in IEEE Double
        NEG_LN_INF = log(3e-324); % The minimum double that log() can calculate
    end

    properties
        sigma = 1; % std of the proposal pdf
        % TotalTimeSteps = 10000; % total iteration
        Ts = 1; % iteration time

        State = [0; randn]; % state at this time, s_{t}

        sample_dim = 1;

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
        function this = Gauss1DV9()
            % Observation specification
            ObservationInfo = rlNumericSpec([2, 1]);
            ObservationInfo.Name = 'Obs';
            ObservationInfo.Description = 's_{t} = [x_{t}; x^{*}_{t+1}]';

            % Action specification
            ActionInfo = rlNumericSpec([4, 1]);
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
            % if isnan(reward)
            %     reward = -this.INF;
            % end
            % if isinf(proposed_sample)
            %     reward = -this.INF;
            % end
        end

        function [Observation, Reward, IsDone, Info] = step(this, Action)
            Info = [];

            % Unpack action
            phi_x_n = Action(1:this.sample_dim+1,:);
            phi_x_n_plus_1 = Action(this.sample_dim+2:end,:);

            mean_phi_x_n = phi_x_n(1:this.sample_dim,:);
            mean_phi_x_n_plus_1 = phi_x_n_plus_1(this.sample_dim+1:end,:);

            std_phi_x_n = phi_x_n(end,:);
            std_phi_x_n_plus_1 = phi_x_n_plus_1(end,:);

            % Unpack state
            x_n = this.State(1);
            x_n_plus_1 = this.State(2);

            % Restore Mean
            mean_n = x_n + mean_phi_x_n;
            mean_n_plus_1 = x_n_plus_1 + mean_phi_x_n_plus_1;

            % Restore Cov
            cov_n = std_phi_x_n^2 * eye(this.sample_dim);
            cov_n_plus_1 = std_phi_x_n_plus_1^2 * eye(this.sample_dim);

            % Calculate Log Target Density
            LogTargetCurrent = this.logTargetPdf(x_n);
            LogTargetProposed = this.logTargetPdf(x_n_plus_1);

            % Calculate Log Proposal Density
            LogProposalCurrent = this.logProposalPdf(x_n, mean_n_plus_1, cov_n);
            LogProposalProposed = this.logProposalPdf(x_n_plus_1, mean_n, cov_n_plus_1);

            % Calculate Log Acceptance Rate
            LogAlphaTemp = LogTargetProposed ...
                - LogTargetCurrent ...
                + LogProposalCurrent ...
                - LogProposalProposed;
            % if isnan(LogAlphaTemp)
            %     LogAlpha = -this.INF;
            % else
                LogAlpha = min( ...
                    0.0, ...
                    LogAlphaTemp ...
                    );
            % end

            % Accept or Reject
            log_u = log(rand());
            if log_u < LogAlpha
                AcceptedStatus = true;
                AcceptedSample = x_n_plus_1;
                AcceptedMean = mean_n_plus_1;
                AcceptedCov = cov_n_plus_1;
            else
                AcceptedStatus = false;
                AcceptedSample = x_n;
                AcceptedMean = mean_n;
                AcceptedCov = cov_n;
            end

            % Update Observation
            % NextProposedSample = mvnrnd(AcceptedMean, AcceptedCov);
            NextProposedSample = normrnd(AcceptedMean, sqrt(AcceptedCov));
            Observation = [AcceptedSample; NextProposedSample];
            this.State = Observation;

            % Calculate Reward
            Reward = this.getReward(x_n, x_n_plus_1, LogAlpha);

            if isinf(Reward) || isnan(Reward)
                fprintf("No. %d\n" + ...
                    "Reward: %f\n" + ...
                    "x_n: %f\n" + ...
                    "mean_n: %f\n" + ...
                    "cov_n: %f\n" + ...
                    "x_n_plus_1: %f\n" + ...
                    "mean_n_plus_1: %f\n" + ...
                    "cov_n_plus_1: %f\n" + ...
                    "LogAlphaTemp: %f\n" + ...
                    "LogTargetProposed: %f\n" + ...
                    "LogTargetCurrent: %f\n" + ...
                    "LogProposalCurrent: %f\n" + ...
                    "LogProposalProposed: %f\n" + ...
                    "LogAlpha: %f\n" + ...
                    "log_u: %f\n", ...
                    this.Ts, ...
                    Reward, ...
                    x_n, ...
                    mean_n, ...
                    cov_n, ...
                    x_n_plus_1, ...
                    mean_n_plus_1, ...
                    cov_n_plus_1, ...
                    LogAlphaTemp, ...
                    LogTargetProposed, ...
                    LogTargetCurrent, ...
                    LogProposalCurrent, ...
                    LogProposalProposed, ...
                    LogAlpha, ...
                    log_u ...
                    );
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
