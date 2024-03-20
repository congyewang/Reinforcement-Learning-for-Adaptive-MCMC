classdef TwinNetworkLayerConstVarDeep < nnet.layer.Layer

    properties (Learnable)
        %{
        Define the learnable parameters of the layer.
        If the structure of nn() is changed, the parameters used in nn that
        need to be optimised need to be declared here.
        %}
        mean_drift_weights_input_hidden1;
        mean_drift_bias_hidden1;
        mean_drift_weights_hidden1_hidden2;
        mean_drift_bias_hidden2;
        mean_drift_weights_hidden2_output;
        mean_drift_bias_output;

        cst_std;
    end

    methods

        function layer = TwinNetworkLayerConstVarDeep(args)
            %{
            This is a constructor function. This function is called first
            at the TwinNetworkLayer instantiate.
            %}
            arguments
                args.Name = "";
                args.input_nodes = []; % dim of x_n
                args.hidden1_nodes = [];
                args.hidden2_nodes = [];
                args.output_nodes = []; % dim of phi(x_n)
            end

            % initialisation
            layer.Name = args.Name;
            layer.mean_drift_weights_input_hidden1 = layer.xavier_uniform_init(args.input_nodes, args.hidden1_nodes);
            layer.mean_drift_bias_hidden1 = zeros(args.hidden1_nodes, 1);
            layer.mean_drift_weights_hidden1_hidden2 = layer.xavier_uniform_init(args.hidden1_nodes, args.hidden2_nodes);
            layer.mean_drift_bias_hidden2 = zeros(args.hidden2_nodes, 1);
            layer.mean_drift_weights_hidden2_output = layer.xavier_uniform_init(args.hidden2_nodes, args.output_nodes);
            layer.mean_drift_bias_output = zeros(args.output_nodes, 1);

            layer.cst_std = 0;
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

        function res = mean_drift_nn(layer, X)
            %{
            Neural network for phi_{mean}(x).
            Args:
                X: input array.
            %}
            X = layer.linear(X, layer.mean_drift_weights_input_hidden1, layer.mean_drift_bias_hidden1);
            X = layer.relu(X);
            X = layer.linear(X, layer.mean_drift_weights_hidden1_hidden2, layer.mean_drift_bias_hidden2);
            X = layer.relu(X);
            X = layer.linear(X, layer.mean_drift_weights_hidden2_output, layer.mean_drift_bias_output);
            res = X;
        end

        function res = predict(layer, observation)
            %{
            Predict function corresponds to mu(s_n) =
            [phi_mean(x_n);phi_mean(x^{*}_{n+1});cst_std].
            Args:
                observation: s_n = [phi(x_n);phi(x^{*}_{n+1})].
            %}
            obs_size = size(observation);
            sample_dim = bitshift(obs_size, -1); % equivalent to sample_dim = obs_size / 2.

            x_n = observation(1:sample_dim, :);
            x_n_plus_1_star = observation(sample_dim+1:end, :);

            mean_drift_n = layer.mean_drift_nn(x_n);
            mean_drift_n_plus_1_star = layer.mean_drift_nn(x_n_plus_1_star);

            the_std = layer.cst_std * ones(1,size(x_n,2));

            res = [mean_drift_n; mean_drift_n_plus_1_star; the_std];
        end

    end
end
