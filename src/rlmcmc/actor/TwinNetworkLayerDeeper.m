classdef TwinNetworkLayerDeeper < nnet.layer.Layer

    properties (Learnable)
        %{
        Define the learnable parameters of the layer.
        If the structure of nn() is changed, the parameters used in nn that
        need to be optimised need to be declared here.
        %}
        weights_input_hidden1;
        bias_hidden1;

        weights_hidden1_hidden2;
        bias_hidden2;

        weights_hidden2_hidden3;
        bias_hidden3;

        weights_hidden3_output;
        bias_output;
    end

    methods

        function layer = TwinNetworkLayerDeeper(args)
            %{
            This is a constructor function. This function is called first
            at the TwinNetworkLayer instantiate.
            %}
            arguments
                args.Name string;
                args.input_nodes int16; % dim of x_n
                args.hidden1_nodes int16;
                args.hidden2_nodes int16;
                args.hidden3_nodes int16;
                args.output_nodes int16; % dim of phi(x_n)
                args.mag double = 1.0;
            end

            % initialisation
            layer.Name = args.Name;

            layer.weights_input_hidden1 = layer.kaiming_normal_init(args.input_nodes, args.hidden1_nodes, args.mag);
            layer.bias_hidden1 = zeros(args.hidden1_nodes, 1);

            layer.weights_hidden1_hidden2 = layer.kaiming_normal_init(args.hidden1_nodes, args.hidden2_nodes, args.mag);
            layer.bias_hidden2 = zeros(args.hidden2_nodes, 1);

            layer.weights_hidden2_hidden3 = layer.kaiming_normal_init(args.hidden2_nodes, args.hidden3_nodes, args.mag);
            layer.bias_hidden3 = zeros(args.hidden2_nodes, 1);

            layer.weights_hidden3_output = layer.kaiming_normal_init(args.hidden3_nodes, args.output_nodes, args.mag);
            layer.bias_output = zeros(args.output_nodes, 1);
        end

        %% initialize Neural Network parameters Function
        function weights = xavier_uniform_init(layer, fan_in, fan_out, mag)
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
            gain = mag * 1.0;
            std = gain * sqrt(2.0 / double(fan_in + fan_out));
            a = sqrt(3.0) * std;
            % Randomly initialise the weight matrix from a uniform distribution U(-a, a)
            weights = unifrnd(-a, a, fan_in, fan_out);
        end

        function weights = xavier_normal_init(layer, fan_in, fan_out, mag)
            %{
            Xavier normal initialisation.

            Args:
                fan_in: Input layer size.
                fan_out: Output layer size.
            %}

            % Calculate standard deviation
            gain = mag * 1.0;
            std = gain * sqrt(2.0 / double(fan_in + fan_out));

            % Randomly initialise the weight matrix using a normal distribution N(0, sigma^2)
            weights = normrnd(0, std, fan_in, fan_out);
        end

        function weights = kaiming_uniform_init(layer, fan_in, fan_out, mag)
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

            % Calculate the range of initialisation bound
            gain = mag * sqrt(2);
            std = gain / sqrt(double(fan_in));
            bound = sqrt(3.0) * std;

            % Randomly initialise the weight matrix from a uniform distribution U(-bound, bound)
            weights = unifrnd(-bound, bound, fan_in, fan_out);
        end

        function weights = kaiming_normal_init(layer, fan_in, fan_out, mag)
            %{
            Kaiming normal initialisation

            Args:
                fan_in: Input layer size.
                fan_out: Output layer size.
            %}

            % Calculate standard deviation
            gain = mag * 1.0;
            std = gain / sqrt(double(fan_in));

            % Randomly initialise the weight matrix using a normal distribution N(0, sigma^2)
            weights = normrnd(0, std, fan_in, fan_out);
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
            Neural network.
            Args:
                X: input array.
            %}
            X = layer.linear(X, layer.weights_input_hidden1, layer.bias_hidden1);
            X = layer.relu(X);
            X = layer.linear(X, layer.weights_hidden1_hidden2, layer.bias_hidden2);
            X = layer.relu(X);
            X = layer.linear(X, layer.weights_hidden2_hidden3, layer.bias_hidden3);
            X = layer.relu(X);
            X = layer.linear(X, layer.weights_hidden3_output, layer.bias_output);
            res = X;
        end

        function res = predict(layer, observation)
            %{
            Predict function corresponds to mu(s_n) =
            [phi(x_n);phi(x^{*}_{n+1})].
            Args:
                observation: s_n = [phi(x_n);phi(x^{*}_{n+1})].
            %}

            [obs_dim, obs_length] = size(observation);
            sample_dim = bitshift(obs_dim, -1); % equivalent to sample_dim = obs_size / 2.
            x_n = observation(1:sample_dim, 1:obs_length);
            x_n_plus_1_star = observation(sample_dim+1:end, 1:obs_length);
            a_n = layer.nn(x_n);
            a_n_plus_1_star = layer.nn(x_n_plus_1_star);
            res = [a_n; a_n_plus_1_star];
        end

    end
end
