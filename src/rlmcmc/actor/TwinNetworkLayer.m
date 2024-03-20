classdef TwinNetworkLayer < nnet.layer.Layer

    properties (Learnable)
        %{
        Define the learnable parameters of the layer.
        If the structure of nn() is changed, the parameters used in nn that
        need to be optimised need to be declared here.
        %}
        weights_input_hidden;
        bias_hidden;
        weights_hidden_output;
        bias_output;
    end

    methods

        function layer = TwinNetworkLayer(args)
            %{
            This is a constructor function. This function is called first
            at the TwinNetworkLayer instantiate.
            %}
            arguments
                args.Name = "";
                args.input_nodes = []; % dim of x_n
                args.hidden_nodes = [];
                args.output_nodes = []; % dim of phi(x_n)
            end
            % initialisation
            layer.Name = args.Name;
            layer.weights_input_hidden = layer.xavier_uniform_init(args.input_nodes, args.hidden_nodes);
            layer.bias_hidden = zeros(args.hidden_nodes, 1);
            layer.weights_hidden_output = layer.xavier_uniform_init(args.hidden_nodes, args.output_nodes);
            layer.bias_output = zeros(args.output_nodes, 1);
        end

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
            a = sqrt(6 / (fan_in + fan_out));
            weights = -a + (2 * a) .* rand(fan_in, fan_out);
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
            X = layer.linear(X, layer.weights_input_hidden, layer.bias_hidden);
            X = layer.relu(X);
            X = layer.linear(X, layer.weights_hidden_output, layer.bias_output);
            res = X;
        end

        function res = predict(layer, observation)
            %{
            Predict function corresponds to mu(s_n) =
            [phi(x_n);phi(x^{*}_{n+1})].
            Args:
                observation: s_n = [phi(x_n);phi(x^{*}_{n+1})].
            %}
            obs_size = size(observation);
            sample_dim = bitshift(obs_size, -1); % equivalent to sample_dim = obs_size / 2.
            x_n = observation(1:sample_dim, :);
            x_n_plus_1_star = observation(sample_dim+1:end, :);
            a_n = layer.nn(x_n);
            a_n_plus_1_star = layer.nn(x_n_plus_1_star);
            res = [a_n; a_n_plus_1_star];
        end

    end
end
