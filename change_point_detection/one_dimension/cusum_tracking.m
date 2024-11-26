%% System Matrices 
F = [0.84 0.23;
     -0.47 0.12];
G = [0.07; 0.23];
C = [1 0];
K = [-1.85 -0.96];

%% Simulation Parameters
T = 2500; % Total timesteps
r = 100; % Buffer size
sigma = 1; % Kernel parameter

change_point = 1500;
burn_in_period = 500;

% Simulate common state process until change point
Z_no_change = zeros(2, T); % State process without change
Z_with_change = zeros(2, T); % State process with change
Y_no_change = zeros(1, T); % Observation process without change
Y_with_change = zeros(1, T); % Observation process with change
u_no_change = zeros(1, T); % Control input without change
u_with_change = zeros(1, T); % Control input with change

% Common noise generation until change point
W_common = mvnrnd(zeros(2,1), 0.01 * eye(2), change_point)'; % Actuation noise W ~ N(0, 0.05I)
V_common = sqrt(0.01) * randn(1, change_point); % Observation noise V ~ N(0, 0.05)

% Generate noise after the change point
W_with_change_after = mvnrnd(zeros(2,1), 0.01 * eye(2), T - change_point)'; % No change after change point
V_with_change_after = sqrt(0.01) * randn(1, T - change_point) + 0.2; % Mean shift after change point
W_no_change_after = mvnrnd(zeros(2,1), 0.01 * eye(2), T - change_point)'; % No change after change point
V_no_change_after = sqrt(0.01) * randn(1, T - change_point); % No change after change point

% Combine noise for complete simulation
W_no_change = [W_common, W_no_change_after];
V_no_change = [V_common, V_no_change_after];
W_with_change = [W_common, W_with_change_after];
V_with_change = [V_common, V_with_change_after];

% Initial state
Z_no_change(:, 1) = [1; 1];
Z_with_change(:, 1) = [1; 1];

% State process simulation
for i = 2:T
    u_no_change(i-1) = K * Z_no_change(:, i-1);
    u_with_change(i-1) = K * Z_with_change(:, i-1);
    Z_no_change(:, i) = F * Z_no_change(:, i-1) + G * u_no_change(i-1) + W_no_change(:, i);
    Z_with_change(:, i) = F * Z_with_change(:, i-1) + G * u_with_change(i-1) + W_with_change(:, i);
end

% Observation process simulation
for i = 1:T
    Y_no_change(:, i) = C * Z_no_change(:, i) + V_no_change(:, i);
    Y_with_change(:, i) = C * Z_with_change(:, i) + V_with_change(:, i);
end

% Set sigma to the median of pairwise distances between samples
all_data = [Y_no_change, Y_with_change];
pairwise_distances = pdist(all_data', 'euclidean');
sigma = median(pairwise_distances);

% MMD computation starting from burn-in period
mmd_values_no_change = zeros(1, floor((T - burn_in_period) / r));
cusum_values_no_change = zeros(1, floor((T - burn_in_period) / r));
cusum_rate_no_change = zeros(1, floor((T - burn_in_period) / r));
mmd_values_with_change = zeros(1, floor((T - burn_in_period) / r));
cusum_values_with_change = zeros(1, floor((T - burn_in_period) / r));
cusum_rate_with_change = zeros(1, floor((T - burn_in_period) / r));

D_h_no_change = Y_no_change(:, burn_in_period+1:burn_in_period + 400); % Reference dataset (1500 timesteps)
D_h_with_change = Y_with_change(:, burn_in_period+1:burn_in_period + 400); % Reference dataset (1500 timesteps)

cusum_no_change = 0;
cusum_with_change = 0;
Delta = 0.005; % Fixed Delta value

for t = (burn_in_period / r + 1):floor(T / r)
    % Update buffer with new window
    B_r_no_change = Y_no_change(:, (t-1)*r + 1:t*r);
    B_r_with_change = Y_with_change(:, (t-1)*r + 1:t*r);
    
    % Compute MMD value for no change
    mmd_values_no_change(t - burn_in_period / r) = compute_MMD(B_r_no_change, D_h_no_change, sigma);
    % Update cusum value for no change
    residual_no_change = mmd_values_no_change(t - burn_in_period / r) - Delta;
    cusum_no_change = cusum_no_change + residual_no_change;
    cusum_no_change = max(0, cusum_no_change); % Reset to zero if negative
    cusum_values_no_change(t - burn_in_period / r) = cusum_no_change;
    % Calculate rate of change for cusum
    if t > (burn_in_period / r + 1)
        cusum_rate_no_change(t - burn_in_period / r) = cusum_values_no_change(t - burn_in_period / r) - cusum_values_no_change(t - burn_in_period / r - 1);
    end
    
    % Compute MMD value for with change
    mmd_values_with_change(t - burn_in_period / r) = compute_MMD(B_r_with_change, D_h_with_change, sigma);
    % Update cusum value for with change
    residual_with_change = mmd_values_with_change(t - burn_in_period / r) - Delta;
    cusum_with_change = cusum_with_change + residual_with_change;
    cusum_with_change = max(0, cusum_with_change); % Reset to zero if negative
    cusum_values_with_change(t - burn_in_period / r) = cusum_with_change;
    % Calculate rate of change for cusum
    if t > (burn_in_period / r + 1)
        cusum_rate_with_change(t - burn_in_period / r) = cusum_values_with_change(t - burn_in_period / r) - cusum_values_with_change(t - burn_in_period / r - 1);
    end
end

% Plot observed Y without and with change point after burn-in period (Updated)
figure;
plot((burn_in_period + 1):T, Y_no_change(1, burn_in_period + 1:end), 'b', 'DisplayName', 'Without Change Point (Component 1)');
hold on;
plot((burn_in_period + 1):T, Y_with_change(1, burn_in_period + 1:end), 'r', 'DisplayName', 'With Change Point (Component 1)');
hold off;
title('Observed Y (Component 1) Without and With Change Point (After Burn-In)');
xlabel('Time Step');
ylabel('Observation Value');
grid on;
legend;

% Plot MMD values without and with change point
figure;
subplot(2, 1, 1);
plot((burn_in_period + r):r:T, mmd_values_no_change, 'b', 'DisplayName', 'MMD Without Change Point');
title('MMD Values Without Change Point');
xlabel('Time Step');
ylabel('MMD Value');
grid on;
legend;

subplot(2, 1, 2);
plot((burn_in_period + r):r:T, mmd_values_with_change, 'r', 'DisplayName', 'MMD With Change Point');
title('MMD Values With Change Point');
xlabel('Time Step');
ylabel('MMD Value');
grid on;
legend;

% Plot CUSUM values without and with change point
figure;
plot((burn_in_period + r):r:T, cusum_values_no_change, 'b', 'DisplayName', 'CUSUM Without Change Point');
hold on;
plot((burn_in_period + r):r:T, cusum_values_with_change, 'r', 'DisplayName', 'CUSUM With Change Point');
hold off;
title('CUSUM Values Without and With Change Point');
xlabel('Time Step');
ylabel('CUSUM Value');
grid on;
legend;

% Plot rate of change of CUSUM without and with change point
figure;
plot((burn_in_period + r):r:T, cusum_rate_no_change, 'b', 'DisplayName', 'Rate of CUSUM Without Change Point');
hold on;
plot((burn_in_period + r):r:T, cusum_rate_with_change, 'r', 'DisplayName', 'Rate of CUSUM With Change Point');
hold off;
title('Rate of Change of CUSUM Without and With Change Point');
xlabel('Time Step');
ylabel('Rate of Change of CUSUM');
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
