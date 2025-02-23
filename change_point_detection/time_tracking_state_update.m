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
r = 500; % Buffer size

change_point = 3000;
burn_in_period = 1000;

% Simulate common state process until change point
Z_no_change = zeros(4, T); % State process without change
Z_with_change = zeros(4, T); % State process with change
Y_no_change = zeros(4, T); % Observation process without change
Y_with_change = zeros(4, T); % Observation process with change

% Common noise generation until change point
W_common = mvnrnd(zeros(4,1), 0.1 * eye(4), change_point)'; % Actuation noise W ~ N(0, 0.1I)
V_common = mvnrnd(zeros(4,1), 0.1 * eye(4), change_point)'; % Observation noise V ~ N(0, 0.1I)

% Generate noise after the change point
W_with_change_after = mvnrnd(0.5 * ones(4,1), 0.1 * eye(4), T - change_point)'; % Mean shift after change point
V_with_change_after = mvnrnd(zeros(4,1), 0.1 * eye(4), T - change_point)'; % Mean shift after change point
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

% Set sigma to the median of pairwise distances between samples
all_data = [Y_no_change, Y_with_change];
pairwise_distances = pdist(all_data', 'euclidean');
sigma = median(pairwise_distances);

% MMD computation starting from burn-in period
mmd_values_no_change = zeros(1, floor((T - burn_in_period) / r));
mmd_values_with_change = zeros(1, floor((T - burn_in_period) / r));

D_h_no_change = Y_no_change(:, burn_in_period+1:burn_in_period + 1500); % Reference dataset (1500 timesteps)
D_h_with_change = Y_with_change(:, burn_in_period+1:burn_in_period + 1500); % Reference dataset (1500 timesteps)

for t = (burn_in_period / r + 1):floor(T / r)
    % Update buffer with new window
    B_r_no_change = Y_no_change(:, (t-1)*r + 1:t*r);
    B_r_with_change = Y_with_change(:, (t-1)*r + 1:t*r);
    
    % Compute MMD value for no change
    mmd_values_no_change(t - burn_in_period / r) = compute_MMD(B_r_no_change, D_h_no_change, sigma);
    
    % Compute MMD value for with change
    mmd_values_with_change(t - burn_in_period / r) = compute_MMD(B_r_with_change, D_h_with_change, sigma);
end

% Plot MMD values without and with change point
figure;

% Plot observed Y without and with change point after burn-in period
subplot(3, 1, 1);
plot((burn_in_period + 1):T, Y_no_change(1, burn_in_period + 1:end), 'b', 'DisplayName', 'Without Change Point (Component 1)');
hold on;
plot((burn_in_period + 1):T, Y_with_change(1, burn_in_period + 1:end), 'r', 'DisplayName', 'With Change Point (Component 1)');
hold off;
title('Observed Y (Component 1) Without and With Change Point (After Burn-In)');
xlabel('Time Step');
ylabel('Observation Value');
grid on;
legend;

subplot(3, 1, 2);
plot((burn_in_period + r):r:T, mmd_values_no_change, 'b', 'DisplayName', 'Without Change Point');
title('MMD Values Without Change Point');
xlabel('Time Step');
ylabel('MMD Value');
grid on;
legend;

subplot(3, 1, 3);
plot((burn_in_period + r):r:T, mmd_values_with_change, 'r', 'DisplayName', 'With Change Point');
title('MMD Values With Change Point');
xlabel('Time Step');
ylabel('MMD Value');
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
