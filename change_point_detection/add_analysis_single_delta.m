%% System Matrices 
A = [0.96, 0.99, -0.88, 0.56;
     0, 0.98, 0.75, -0.65;
     0, 0, 0.97, 0.95;
     0, 0, 0, 0.94];
C = [1, 0, 0, 0;
     0, 0, 0, 0;
     0, 0, 1, 0;
     0, 0, 0, 0];

%% Simulation Parameters
T = 10000; % Total timesteps
r = 100; % Buffer size
sigma = 1; % Kernel parameter

change_point = 3000;
burn_in_period = 1000;
Delta = 0.05; % Fixed Delta value for this experiment

% Simulate common state process until change point
Z_no_change = zeros(4, T); % State process without change
Z_with_change = zeros(4, T); % State process with change
Y_no_change = zeros(4, T); % Observation process without change
Y_with_change = zeros(4, T); % Observation process with change

% Common noise generation until change point
W_common = mvnrnd(zeros(4,1), 0.1 * eye(4), change_point)'; % Actuation noise W ~ N(0, 0.1I)
V_common = mvnrnd(zeros(4,1), 0.1 * eye(4), change_point)'; % Observation noise V ~ N(0, 0.1I)

% Generate noise after the change point (variance change)
W_with_change_after = mvnrnd(zeros(4,1), 0.5 * eye(4), T - change_point)'; % Variance change after change point
V_with_change_after = mvnrnd(zeros(4,1), 0.5 * eye(4), T - change_point)'; % Variance change after change point
W_no_change_after = mvnrnd(zeros(4,1), 0.1 * eye(4), T - change_point)'; % No change after change point
V_no_change_after = mvnrnd(zeros(4,1), 0.1 * eye(4), T - change_point)'; % No change after change point

% Combine noise for complete simulation
W_no_change = [W_common, W_no_change_after];
V_no_change = [V_common, V_no_change_after];
W_with_change = [W_common, W_with_change_after];
V_with_change = [V_common, V_with_change_after];

% Initial state
Z_no_change(:, 1) = [1; 1; 1; 1];
Z_with_change(:, 1) = [1; 1; 1; 1];

% State process simulation
for i = 2:T
    Z_no_change(:, i) = A * Z_no_change(:, i-1) + W_no_change(:, i);
    Z_with_change(:, i) = A * Z_with_change(:, i-1) + W_with_change(:, i);
end

% Observation process simulation
for i = 1:T
    Y_no_change(:, i) = C * Z_no_change(:, i) + V_no_change(:, i);
    Y_with_change(:, i) = C * Z_with_change(:, i) + V_with_change(:, i);
end

% Initialize CUSUM test variables
cusum_no_change = zeros(1, floor((T - burn_in_period) / r));
cusum_with_change = zeros(1, floor((T - burn_in_period) / r));

s_t_no_change = 0;
s_min_no_change = 0;
s_t_with_change = 0;
s_min_with_change = 0;

D_h_no_change = Y_no_change(:, burn_in_period:burn_in_period + 199); % Reference dataset (200 timesteps)
D_h_with_change = Y_with_change(:, burn_in_period:burn_in_period + 199); % Reference dataset (200 timesteps)

for t = (burn_in_period / r + 3):floor(T / r) % Start from the 3rd window after burn-in
    % Update buffer with new window
    B_r_no_change = Y_no_change(:, (t-1)*r + 1:t*r);
    B_r_with_change = Y_with_change(:, (t-1)*r + 1:t*r);
    
    % Compute MMD value for no change
    MMD_value_no_change = compute_MMD(B_r_no_change, D_h_no_change, sigma);
    residual_no_change = MMD_value_no_change - Delta;
    s_t_no_change = s_t_no_change + residual_no_change;
    s_min_no_change = min(s_t_no_change, s_min_no_change);
    cusum_no_change(t - burn_in_period / r) = s_t_no_change;
    
    % Compute MMD value for with change
    MMD_value_with_change = compute_MMD(B_r_with_change, D_h_with_change, sigma);
    residual_with_change = MMD_value_with_change - Delta;
    s_t_with_change = s_t_with_change + residual_with_change;
    s_min_with_change = min(s_t_with_change, s_min_with_change);
    cusum_with_change(t - burn_in_period / r) = s_t_with_change;
end

% Plot CUSUM values without and with change point
figure;
subplot(2, 1, 1);
plot((burn_in_period + 3*r):r:T, cusum_no_change((3:end)), 'b', 'DisplayName', 'CUSUM Without Change Point');
title('CUSUM Values Without Change Point');
xlabel('Time Step');
ylabel('CUSUM Value');
grid on;
legend;

subplot(2, 1, 2);
plot((burn_in_period + 3*r):r:T, cusum_with_change((3:end)), 'r', 'DisplayName', 'CUSUM With Change Point');
title('CUSUM Values With Change Point');
xlabel('Time Step');
ylabel('CUSUM Value');
grid on;
legend;

%% MMD computation function
function mmd_value = compute_MMD(B_r, D_h, sigma)
    r = size(B_r, 2);
    h = size(D_h, 2);
    
    % Compute the three terms for MMD
    term1 = 0;
    for i = 1:r
        for j = 1:r
            term1 = term1 + kernel_rq(B_r(:,i), B_r(:,j), sigma);
        end
    end
    term1 = term1 / r^2;
    
    term2 = 0;
    for i = 1:h
        for j = 1:h
            term2 = term2 + kernel_rq(D_h(:,i), D_h(:,j), sigma);
        end
    end
    term2 = term2 / h^2;
    
    term3 = 0;
    for i = 1:r
        for j = 1:h
            term3 = term3 + kernel_rq(B_r(:,i), D_h(:,j), sigma);
        end
    end
    term3 = term3 / (r * h);
    
    % Compute MMD value
    mmd_value = sqrt(term1 + term2 - 2 * term3);
end

%% Rational quadratic kernel function
function k_val = kernel_rq(x, y, sigma)
    k_val = (1 + norm(x - y)^2 / (2 * sigma))^(-sigma);
end
