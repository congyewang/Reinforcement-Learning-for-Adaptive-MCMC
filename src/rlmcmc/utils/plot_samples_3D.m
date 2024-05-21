function plot_samples_3D(sample, proposed_sample, log_target_pdf, color_list, lb, ub)
if nargin < 4
    color_list = [1,0,0,0.1];
end
if nargin < 5
    lb = -6;
end
if nargin < 6
    ub = 6;
end

hold on;
for i = 1:size(sample, 1)
    create_mean_drift_3D(sample(i,:), proposed_sample(i,:), log_target_pdf, color_list, lb, ub)
end

hold off;
view(3);

end

function create_mean_drift_3D(sample, proposed_sample, log_target_pdf, color_list, lb, ub)
if nargin < 4
    color_list = [1,0,0,0.1];
end
if nargin < 5
    lb = -6;
end
if nargin < 6
    ub = 6;
end

plot_target_3D(log_target_pdf, lb, ub);

arrow_dx = proposed_sample(1)- sample(1);
arrow_dy = proposed_sample(2)- sample(2);
% arrow_dz = -sqrt(arrow_dx^2 + arrow_dy^2 + 3^2);
% arrow_dz = -3;
arrow_dz = -1;

% Plot the arrow from 'sample' to 'mean' in 3D
hold on;
% quiver3(sample(1), sample(2), 1.5, arrow_dx, arrow_dy, arrow_dz, 'off', 'MaxHeadSize', 0.5, 'Color', 'r', 'LineWidth', 1.5);
quiver3(sample(1), sample(2), 0.5, arrow_dx, arrow_dy, arrow_dz, 'off', 'MaxHeadSize', 0.5, 'Color', color_list, 'LineWidth', 0.5);

hold off;

end

function plot_target_3D(log_target_pdf, lb, ub)

if nargin < 2
    lb = -6;
end
if nargin < 3
    ub = 6;
end

[X1, X2, X3] = meshgrid(lb:.1:ub);
X = [X1(:) X2(:)];

p = log_target_pdf(X);
P = reshape(p, size(X1));

% zslice = [-1.5,1.5];
zslice = [-0.5,0.5];
contourslice(X1,X2,X3,P,[],[], zslice);

end
