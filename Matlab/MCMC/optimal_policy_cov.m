function sigma2 = optimal_policy_cov(x, theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10, theta11, theta12)
    if nargin < 14
        theta0 = 0;
        theta1 = 0;
        theta2 = 0;
        theta3 = 1;
        theta4 = 2.5;
        theta5 = 0;
        theta6 = 2.5;
        theta7 = 0;
        theta8 = 1;
        theta9 = 2.5;
        theta10 = 0;
        theta11 = 2.5;
        theta12 = 0;
    end

    phi = acos(theta0 + theta1*x(1) + theta2*x(2));
    alpha = theta3^2 + theta4^2 * (x(1) - theta5)^2 + theta6^2 * (x(2) - theta7)^2;
    beta = theta8^2 + theta9^2 * (x(1) - theta10)^2 + theta11^2 * (x(2) - theta12)^2;

    t1 = [cos(phi), -sin(phi); sin(phi), cos(phi)];
    t2 = [alpha, 0; 0, beta];

    sigma2 = t1 * t2 * t1';

    return
end
