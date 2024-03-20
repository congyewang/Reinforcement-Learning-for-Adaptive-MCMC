classdef TwinNetworkLayerDeep < nnet.layer.Layer
    %% Weights and Bias
    properties (Learnable)
        %{
        Define the learnable parameters of the layer.
        One thing to keep in mind is to be sure to add `(Learnable)`!

        If the structure of nn() is changed, the parameters used in nn that
        need to be optimised need to be declared here.
        %}
        weights_input_hidden1;
        bias_hidden1;

        weights_hidden1_hidden2;
        bias_hidden2;

        weights_hidden2_output;
        bias_output;

        % alpha; % Only Used in prelu activation function
    end

    methods
        % !!! Note that all the first parameter within the methods, layer
        % is a pointer, which cannot be deleted. Not recommended omitted
        % using ~.

        %% Constructor Function
        function layer = TwinNetworkLayerDeep(args)
            %{
            This is a constructor function. This function is called first
            at the TwinNetworkLayer instantiate.
            %}

            % Default ararguments
            arguments
                % Name of the Layer
                args.Name = "";
                % The number of the unit in the input layer , which should be equal to
                % the dim of the sample, x_n.
                args.input_nodes = 2;
                % The number of the unit in the hidden layer. Generally a
                % multiple of 2 or a power of 2.
                args.hidden1_nodes = 8;
                args.hidden2_nodes = 8;
                % The number of the unit in the input layer , which should be equal to
                % the dim of the action corresponding to x_n.
                args.output_nodes = 2;
            end

            % Set layer name.
            layer.Name = args.Name;
            % Set layer Description. (optional)
            layer.Description = "Custom layer with two fully connected sublayers and ReLU activation";

            % Initial Weights and Bias
            layer.weights_input_hidden1 = layer.xavier_uniform_init(args.input_nodes, args.hidden1_nodes);
            layer.bias_hidden1 = zeros(args.hidden1_nodes, 1);

            layer.weights_hidden1_hidden2 = layer.xavier_uniform_init(args.hidden1_nodes, args.hidden2_nodes);
            layer.bias_hidden2 = zeros(args.hidden2_nodes, 1);

            layer.weights_hidden2_output = layer.xavier_uniform_init(args.hidden2_nodes, args.output_nodes);
            layer.bias_output = zeros(args.output_nodes, 1);

            % Only Used in prelu activation function
            % layer.alpha = 0.25;
        end

        %% initialize Neural Network parameters Function
        function weights = xavier_uniform_init(layer, fan_in, fan_out)
            %{
            Xavier uniform initialisation. The purpose of the Xavier
            initialisation is to keep the variance of the inputs constant
            at each layer while training the deep network to avoid the
            problem of vanishing or exploding gradients.

            Args:
                fan_in: Input layer size.
                fan_out: Output layer size.
            %}

            % Calculate the range of initialisation a
            a = sqrt(6 / (fan_in + fan_out));

            % Randomly initialise the weight matrix from a uniform distribution U(-a, a)
            weights = -a + (2 * a) .* rand(fan_in, fan_out);
        end

        function weights = xavier_normal_init(layer, fan_in, fan_out)
            %{
            Xavier normal initialisation.

            Args:
                fan_in: Input layer size.
                fan_out: Output layer size.
            %}

            % Calculate standard deviation
            sigma = sqrt(2 / (fan_in + fan_out));

            % Randomly initialise the weight matrix using a normal distribution N(0, sigma^2)
            weights = sigma * randn(fan_in, fan_out);
        end

        function weights = kaiming_uniform_init(layer, fan_in, fan_out)
            %{
            Kaiming uniform initialisation, which is a weight initialisation
            method specifically designed to solve the problem of gradient
            vanishing/exploding of ReLU activation functions in deep neural
            networks. This method assumes that the activation function is a
            ReLU and therefore takes into account the properties of ReLU
            during initialisation.

            Args:
                fan_in: Input layer size.
                fan_out: Output layer size.
            %}

            % Calculate the range of initialisation a
            a = sqrt(6 / fan_in);

            % Randomly initialise the weight matrix from a uniform distribution U(-a, a)
            weights = -a + (2 * a) .* rand(fan_in, fan_out);
        end

        function weights = kaiming_normal_init(layer, fan_in, fan_out)
            %{
            Kaiming normal initialisation

            Args:
                fan_in: Input layer size.
                fan_out: Output layer size.
            %}

            % Calculate standard deviation
            sigma = sqrt(2 / fan_in);

            % Randomly initialise the weight matrix using a normal distribution N(0, sigma^2)
            weights = sigma * randn(fan_in, fan_out);
        end

        %% Non Linear Activation Function
        function res = relu(layer, x)
            %{
            ReLU Activation Function

            Args:
                x: Input array.
            %}
            res = max(0, x);
        end

        function res = elu(layer, x, alpha)
            %{
            ELU Activation Function.

            Args:
                x: Input array.
                alpha: the alpha value for the ELU formulation. Default: 1.0
            %}
            if nargin < 3
                alpha = 1.0;
            end

            res = x; % initialise output
            for i = 1:numel(x)
                if x(i) <= 0
                    res(i) = alpha * (exp(x(i)) - 1);
                end
                % when x > 0, y(i) = x(i), no change required.
            end
        end

        function res = tanh(layer, x)
            %{
            Tanh Activation Function (element-wise)

            Args:
                x: Input array.
            %}
            res = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
        end

        function res = sigmoid(layer, x)
            %{
            Sigmoid Activation Function (element-wise)

            Args:
                x: Input array.
            %}
            res = 1 ./ (1 + exp(-x));
        end

        function res = hardtanh(layer, x, min_val, max_val)
            %{
            Hardtanh Activation Function.

            Args:
              x: Input array.
            %}
            if nargin < 3
                min_val = -1.0;
            end
            if nargin < 4
                max_val = 1.0;
            end

            res = max(min_val, min(max_val, x));
        end

        function res = hardswish(layer, x)
            %{
            HardSwish Activation Function.

            Args:
                x: Input array.
            %}

            % Calculate ReLU6
            relu6 = min(max(x + 3, 0), 6);

            % Calculate HardSwish
            res = x .* (relu6 / 6);
        end

        function res = relu6(layer, x)
            %{
            ReLU6 Activation Function.

            Args:
                x: Input array.
            %}

            % Use the max function to set all negative values in x to 0
            res = max(0, x);

            % Use the min function to set all values in y greater than 6 to 6
            res = min(res, 6);
        end

        function res = selu(layer, x)
            %{
            SeLU Activation Function.

            Args:
                x: Input array.
            %}
            alpha = 1.6732632423543772848170429916717;
            lambda = 1.0507009873554804934193349852946;

            % Apply lambda * x to positive values of x
            res_pos = lambda * x .* (x > 0);

            % Apply lambda * alpha * (exp(x) - 1) to negative values of x.
            res_neg = lambda * alpha * (exp(x) - 1) .* (x <= 0);

            % Results of combining positive and negative components
            res = res_pos + res_neg;
        end

        function res = celu(layer, x, alpha)
            %{
            CeLU Activation Function.

            Args:
                x: Input array.
                alpha: Parameters of CELU, controlling the shape of the 
                       negative part.
            %}

            if nargin < 3
                alpha = 1.0;
            end
            % Verify that alpha is positive
            if alpha <= 0
                error('alpha must be positive for CELU function.');
            end

            % Returns x directly for the positive part of x
            res_pos = x .* (x > 0);

            % Applying the CELU formula to the negative part of x
            res_neg = alpha * (exp(x / alpha) - 1) .* (x <= 0);

            % Results of combining positive and negative components
            res = res_pos + res_neg;
        end

        function res = leaky_relu(layer, x, alpha)
            %{
            Leaky ReLU Activation Function.

            Args:
                x: Input array.
                alpha: Slope of the negative part, a very small positive
                constant. (Default alpha = 0.01)
            %}
            if nargin < 3
                alpha = 0.01;
            end
            % Returns x directly for positive parts of x, applies
            % alpha * x for negative parts of x.
            res = max(x, alpha * x);
        end

        function res = prelu(layer, x, alpha)
            %{
            PReLU Activation Function.

            Args:
                x: Input array.
                alpha: Slope of the negative part, a very small positive
                constant. (Default alpha = 0.01)
            %}

            % Returns x directly for positive parts of x, applies
            % alpha * x for negative parts of x.
            res = max(x, alpha .* x);
        end

        function res = rrelu(layer, x, l, u, isTraining)
            %{
            RReLU Activation Function.

            Args:
                x: Input array.
                l: Lower limit of alpha.
                u: Upper limit of alpha.
                isTraining: Bool, indicating whether it is in the
                            training phase.
            %}

            % Calculate alpha
            if nargin < 3
                l = 1.0 / 8;
            end
            if nargin < 4
                u = 1.0 / 3;
            end
            if nargin < 5
                isTraining = true;
            end

            if isTraining
                % Training phase: randomly select alpha from the range
                % [l, u].
                alpha = (l + (u - l) .* rand(size(x)));
            else
                % Test phase: Use the average of the alpha.
                alpha = (l + u) / 2;
            end

            res = max(x, alpha .* x);
        end

        function res = glu(layer, x)
            %{
            GLU Activation Function.

            Args:
                x: Input array, which should be even length vectors or even
                   columns of other dimension matrices.
            %}

            % Check the input size to make sure it can be split evenly.
            [rows, cols] = size(x);
            if mod(cols, 2) ~= 0
                error('Input size must be even for GLU.');
            end

            % Split the input into two equal parts.
            a = x(:, 1:cols/2);
            b = x(:, cols/2+1:end);

            % Apply Sigmoid function to b and element-wise multiply.
            res = a .* layer.sigmoid(b);
        end

        function res = gelu(layer, x)
            %{
            GELU Activation Function.

            Args:
                x: Input array.
            %}

            res = 0.5 * x .* (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x.^3)));
        end

        function res = logsigmoid(layer, x)
            %{
            Log Sigmoid Activation Function.

            Args:
                x: Input array.
            %}

            res = log(layer.sigmoid(x));
        end

        function res = softsign(layer, x)
            %{
            Softsign Activation Function.

            Args:
                x: Input array.
            %}
            res = x ./ (1 + abs(x));
        end

        function res = softplus(layer, x)
            %{
            Softplus Activation Function.

            Args:
                x: Input array.
            %}

            res = log(1 + exp(x));
        end

        function res = softmax(layer, x)
            %{
            Softmax Activation Function.

            Args:
                x: Input array.
            %}

            exps = exp(x - max(x, [], 2));
            res = exps ./ sum(exps, 2);
        end

        function res = softmin(layer, x)
            %{
            Softmin Activation Function.

            Args:
                x: Input array.
            %}

            res = layer.softmax(-x);
        end

        function res = gumbel_softmax(layer, x, tau)
            %{
            Gumbel-Softmax Activation Function.

            Args:
                x: Input array.
                tau: Temperature parameter that controls the sharpness of the output distribution
            %}

            if nargin < 3
                tau = 1.0;
            end
            % Sample from a uniform distribution U(0,1)
            u = rand(size(x));

            % Calculate Gumbel noise and add to logic values
            g = -log(-log(u));
            gumbel_logits = x + g;

            % Applying the Softmax function
            res = layer.softmax(gumbel_logits / tau);
        end

        function res = silu(layer, x)
            %{
            SiLU (Swish) Activation Function.

            Args:
                x: Input array.
            %}

            res = x .* layer.sigmoid(x);
        end

        function res = mish(layer, x)
            %{
            Mish Activation Function.

            Args:
                x: Input array.
            %}

            res = x .* layer.tanh(layer.softplus(x));
        end

        function res = normalise(layer, x)
            %{
            Normalise Function.

            Args:
               x: Input array.
            %}

            % Calculate the number of Euclidean paradigms for each column
            norms = sqrt(sum(x.^2, 1));

            % Avoid division by zero
            norms(norms == 0) = 1;

            % Normalise each column
            res = x ./ norms;
        end

        %% Linear Function
        function res = linear(layer, X, weights, bias)
            %{
            Linear Function in a Layer.

            Args:
                X: Input array.
                weights: weights array initialised from Constructor Function.
                bias: bias array initialised from Constructor Function.
            %}
            res = weights' * X + bias;
        end

        function res = nn(layer, X)
            %{
            Neural Network, which corresponds to mu(x).

            Args:
                X: input array.
            %}
            X = layer.linear(X, layer.weights_input_hidden1, layer.bias_hidden1);
            X = layer.relu(X);
            X = layer.linear(X, layer.weights_hidden1_hidden2, layer.bias_hidden2);
            X = layer.relu(X);
            X = layer.linear(X, layer.weights_hidden2_output, layer.bias_output);
            res = X;
        end

        function res = predict(layer, observation)
            %{
            Predict function is like the forward function but without
            gradient backpropagation, which is used in the infer phase
            after training. It corresponds to mu(s_n) =
            [phi(x_n);phi(x^{*}_{n+1})].
            
            Args:
                observation: s_n = [phi(x_n);phi(x^{*}_{n+1})].
            %}
            obs_size = size(observation);
            % To keep the sample_dim as a integers, use Shift bits
            % operation. Equivalent to sample_dim = obs_size / 2.
            sample_dim = bitshift(obs_size, -1);

            x_n = observation(1:sample_dim, :);
            x_n_plus_1_star = observation(sample_dim+1:end, :);

            a_n = layer.nn(x_n);
            a_n_plus_1_star = layer.nn(x_n_plus_1_star);

            res = [a_n; a_n_plus_1_star];
        end

    end
end