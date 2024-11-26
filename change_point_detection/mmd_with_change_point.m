% system matrices
A = [0.96, 0.99, -0.88, 0.56;
     0, 0.98, 0.75, -0.65;
     0, 0, 0.97, 0.95;
     0, 0, 0, 0.94];
C = [1, 0, 0, 0;
     0, 0, 0, 0;
     0, 0, 1, 0;
     0, 0, 0, 0];

% sys params
num_simulations = 50;
T = 1e3;
change_point = 500;
sigma = 2;
r = 50;


thresholds = 0.01:0.05:1;
Delta_vals = [0.15, 0.05, 0.1];

ADD_results = zeros(length(thresholds), length(Delta_vals));

for sim_idx = 1:num_simulations
    fprintf('Simulation %d\n', sim_idx);
    
    % initial W, V: before change point
    W = mvnrnd(zeros(4,1), eye(4), T)';
    V = mvnrnd(zeros(4,1), eye(4), T)';

    % state process, observed output variable
    Z = zeros(4, T);
    Y = zeros(4, T);
    Z(:, 1) = [1; 1; 1; 1];
    

    for i = 2:T
        if i > change_point
            % mean shift beyond change point timestep
            W(:, i) = mvnrnd(0.011 * ones(4, 1), eye(4), 1)';
            V(:, i) = mvnrnd(0.011 * ones(4, 1), eye(4), 1)';
        end
        Z(:, i) = A * Z(:, i-1) + W(:, i);
        Y(:, i) = C * Z(:, i) + V(:, i);
    end
    
    % reference dataset 1000 steps
    D_h = Y(:, 1:1000);
    
    % iterate over all delta values
    for j = 1:length(Delta_vals)
        Delta = Delta_vals(j);
        delays = zeros(length(thresholds), T);
        % delays to store the delay for each threshold
        % initialize with T: no alarm situation
        
        for idx = 1:length(thresholds)
            b = thresholds(idx);
            s_t = 0;
            s_min = 0;
            detected = false;
            
            % going over each buffer window
            for t = 1:floor(T/r)
                B_r = Y(:, (t-1)*r + 1:t*r);
                
                MMD_value = compute_MMD(B_r, D_h, sigma);
                
                s_t = s_t + MMD_value - Delta;
                s_min = min(s_t, s_min);
                
                if (s_t - s_min) > b && t * r > change_point
                    delays(idx) = t * r - change_point;
                    fprintf('Detection: Threshold=%.3f, Delta=%.3f, MMD=%.4f, s_t=%.4f, Delay=%d at time %d\n', ...
                            b, Delta, MMD_value, s_t, delays(idx), t * r);
                    detected = true;
                    break;
                end
            end
            
            if ~detected
                fprintf('No detection for Threshold=%.3f, Delta=%.3f\n', b, Delta);
            end
        end
        ADD_results(:, j) = ADD_results(:, j) + delays / num_simulations;
    end
end

figure;
for j = 1:length(Delta_vals)
    plot(thresholds, ADD_results(:, j), '-o', 'Color', colors(j), 'MarkerFaceColor', colors(j), ...
         'DisplayName', sprintf('ADD \\Delta = %.3f', Delta_vals(j)));
    hold on;
end
title('Average ADD vs Threshold for Different Delta Values');
xlabel('Threshold');
ylabel('ADD');
legend;
grid on;
hold off;
