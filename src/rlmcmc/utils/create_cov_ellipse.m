function create_cov_ellipse(covariance, mean, sample)
    msd = 1.0;
 
    [eigVecs, eigVals] = eig(covariance);
    eigVals = diag(eigVals);  % Extract the eigenvalues into a vector

    % Calculate the angle of the ellipse
    angle = atan2(eigVecs(2, 1), eigVecs(1, 1));
    angle = rad2deg(angle);

    % Calculate the width and height of the ellipse
    width = msd * sqrt(eigVals(1));
    height = msd * sqrt(eigVals(2));

    % Plot the ellipse
    ellipse(width, height, angle, mean(1), mean(2));

    % Check if the 'sample' argument is provided and plot an arrow if it is
    if nargin >= 3
        % Calculate the direction for the arrow
        arrow_dx = mean(1) - sample(1);
        arrow_dy = mean(2) - sample(2);

        % Plot the arrow from 'sample' to 'mean'
        hold on;
        quiver(sample(1), sample(2), arrow_dx, arrow_dy, 0, 'MaxHeadSize', 0.5, 'Color', 'r', 'LineWidth', 1.5);
        hold off;
    end
end

function ellipse(ra, rb, ang, x0, y0)
    % This function plots an ellipse with semimajor axis of 'ra', semiminor axis 'rb',
    % rotation 'ang' in degrees, centered at (x0, y0).
    
    hold on;
    ang = deg2rad(ang);
    th = 0:pi/50:2*pi;
    x = ra * cos(th) * cos(ang) - rb * sin(th) * sin(ang) + x0;
    y = ra * cos(th) * sin(ang) + rb * sin(th) * cos(ang) + y0;

    plot(x, y, 'b', 'LineWidth', 1.5);
    axis equal;
    grid on;
    hold off;
end