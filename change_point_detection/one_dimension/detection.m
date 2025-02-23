%% System Matrices 
F = [0.84 0.23;
     -0.47 0.12];
G = [0.07; 0.23];
C = [1 0];
K = [-1.85 -0.96];

%% Simulation Parameters
T = 2250; % Total timesteps
r = 100; % Buffer size

change_point = 1500;
burn_in_period = 500;
Deltas = [0.001, 0.005, 0.01]; % Different Delta values
thresholds = 0.8:0.02:1.5; % Range of thresholds to test
num_repeats = 3; % Number of repetitions for averaging detection delay and rate

% Store detection delays and rates
avg_detection_delays = zeros(length(thresholds), length(Deltas));
detection_rates = zeros(length(thresholds), length(Deltas));

for delta_idx = 1:length(Deltas)
    Delta = Deltas(delta_idx);
    fprintf('Simulating for Delta = %f\n', Delta);
    detection_delays = zeros(length(thresholds), num_repeats); % Store detection delays for each repeat
    detections = zeros(length(thresholds), num_repeats); % Store detection status for each repeat

    for repeat = 1:num_repeats
        fprintf('Repeat %d for Delta = %f\n', repeat, Delta);
        % Reinitialize state and observation processes for each repeat
        Z_no_change = zeros(2, T); % State process without change
        Z_with_change = zeros(2, T); % State process with change
        Y_no_change = zeros(1, T); % Observation process without change
        Y_with_change = zeros(1, T); % Observation process with change
        u_no_change = zeros(1, T); % Control input without change
        u_with_change = zeros(1, T); % Control input with change

        % Common noise generation until change point
        W_common = mvnrnd(zeros(2,1), 0.01 * eye(2), change_point)'; % Actuation noise W ~ N(0, 0.01I)
        V_common = sqrt(0.01) * randn(1, change_point); % Observation noise V ~ N(0, 0.01)

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

        % Reference dataset initialization
        D_h_no_change = Y_no_change(:, burn_in_period+1:burn_in_period + 400); % Reference dataset (1500 timesteps)
        D_h_with_change = Y_with_change(:, burn_in_period+1:burn_in_period + 400); % Reference dataset (1500 timesteps)

        % MMD and CUSUM computation starting from burn-in period
        s_t_no_change = 0;
        s_t_with_change = 0;
        s_min_no_change = 0;
        s_min_with_change = 0;

        thresholds_crossed = false(1, length(thresholds));
        detection_times = NaN(1, length(thresholds));

        for t = (change_point / r + 1):floor(T / r)
            % Update buffer with new window
            B_r_no_change = Y_no_change(:, (t-1)*r + 1:t*r);
            B_r_with_change = Y_with_change(:, (t-1)*r + 1:t*r);
            
            % Compute MMD value for no change
            MMD_value_no_change = compute_MMD(B_r_no_change, D_h_no_change, sigma);
            residual_no_change = MMD_value_no_change - Delta;
            s_t_no_change = s_t_no_change + residual_no_change;
            s_min_no_change = min(s_t_no_change, s_min_no_change);
            cusum_no_change = s_t_no_change;
            
            % Compute MMD value for with change
            MMD_value_with_change = compute_MMD(B_r_with_change, D_h_with_change, sigma);
            residual_with_change = MMD_value_with_change - Delta;
            s_t_with_change = s_t_with_change + residual_with_change;
            s_min_with_change = min(s_t_with_change, s_min_with_change);
            cusum_with_change = s_t_with_change;

            % Check for detection for each threshold
            for idx = 1:length(thresholds)
                threshold = thresholds(idx);
                if ~thresholds_crossed(idx) && (s_t_with_change - s_min_with_change) > threshold
                    detection_times(idx) = t * r - change_point;
                    thresholds_crossed(idx) = true;
                end
            end

            % Break if all thresholds are crossed
            if all(thresholds_crossed)
                break;
            end
        end

        % Store detection delays and rate for this repeat
        for idx = 1:length(thresholds)
            if ~isnan(detection_times(idx)) && detection_times(idx) > 0
                detection_delays(idx, repeat) = detection_times(idx);
                detections(idx, repeat) = 1;
            else
                detection_delays(idx, repeat) = T - change_point; % Set maximum delay if not detected
                detections(idx, repeat) = 0;
            end
        end
    end

    % Average detection delay across repetitions
    avg_detection_delays(:, delta_idx) = mean(detection_delays, 2);
    detection_rates(:, delta_idx) = mean(detections, 2);
end

% Plot detection rate vs threshold for different Delta values
figure;
hold on;
colors = ['r', 'g', 'b'];
markers = ['o', 's', '^'];
for delta_idx = 1:length(Deltas)
    plot(thresholds, detection_rates(:, delta_idx), ['-' markers(delta_idx)], 'Color', colors(delta_idx), ...
         'DisplayName', sprintf('\\Delta = %.3f', Deltas(delta_idx)), 'MarkerSize', 4, 'LineWidth', 1);
end
hold off;
title('Detection Rate vs Threshold for Different \Delta Values');
xlabel('Threshold');
ylabel('Detection Rate');
grid on;
legend('Location', 'best');

% Plot average detection delay vs threshold for different Delta values
figure;
hold on;
for delta_idx = 1:length(Deltas)
    plot(thresholds, avg_detection_delays(:, delta_idx), ['-' markers(delta_idx)], 'Color', colors(delta_idx), ...
         'DisplayName', sprintf('\\Delta = %.3f', Deltas(delta_idx)), 'MarkerSize', 4, 'LineWidth', 1);
end
hold off;
title('Average Detection Delay vs Threshold for Different \Delta Values');
xlabel('Threshold');
ylabel('Average Detection Delay');
grid on;
legend('Location', 'best');

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
