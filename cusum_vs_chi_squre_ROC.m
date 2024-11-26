num_simulations = 1000;
num_timesteps = 150;
attack_start = 100;
attack_magnitude = 2;
tau_values = linspace(7, 4, 70); % Threshold values for CUSUM
chi_thresholds = linspace(15, 10, 70); % Threshold values for chi-squared
bias = 1.5; % Fixed bias for CUSUM

% System matrices
F = [0.84 0.23; -0.47 0.12];
G = [0.07; 0.23];
C = [1 0];
K = [-1.85 -0.96];
R1 = [0.45 -0.11; -0.11 0.20];
R2 = 1;
L = [0.31; -0.21];
Sigma = [1.70];

% Initialize performance metrics
Pfa_cusum = zeros(size(tau_values));
Pd_cusum = zeros(size(tau_values));
Pfa_chi = zeros(size(chi_thresholds));
Pd_chi = zeros(size(chi_thresholds));

% Initialize counters for false alarms and detections for both detectors
false_alarms_cusum = zeros(size(tau_values));
detections_cusum = zeros(size(tau_values));
false_alarms_chi = zeros(size(chi_thresholds));
detections_chi = zeros(size(chi_thresholds));

% Simulations without attacks (for false alarms)
for sim = 1:num_simulations
    [fa_cusum, fa_chi] = run_simulation(tau_values, chi_thresholds, false, ...
        num_timesteps, attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L, Sigma);
    false_alarms_cusum = false_alarms_cusum + fa_cusum;
    false_alarms_chi = false_alarms_chi + fa_chi;
end

% Simulations with attacks (for detections)
for sim = 1:num_simulations
    [det_cusum, det_chi] = run_simulation(tau_values, chi_thresholds, true, ...
        num_timesteps, attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L, Sigma);
    detections_cusum = detections_cusum + det_cusum;
    detections_chi = detections_chi + det_chi;
end

% Calculate probabilities for CUSUM and chi-squared
Pfa_cusum = false_alarms_cusum / num_simulations;
Pd_cusum = detections_cusum / num_simulations;
Pfa_chi = false_alarms_chi / num_simulations;
Pd_chi = detections_chi / num_simulations;

% ROC curve for both detectors
figure;
plot(Pfa_cusum, Pd_cusum, 'b-o', 'LineWidth', 1.5); hold on;
plot(Pfa_chi, Pd_chi, 'r-s', 'LineWidth', 1.5);
xlabel('Probability of False Alarm (Pfa)');
ylabel('Probability of Detection (Pd)');
title('ROC Curve for CUSUM and Chi-Square Detectors');
legend('CUSUM', 'Chi-Squared');
grid on;

% Markers for tau values (CUSUM)
for i = 1:2:length(tau_values)
    text(Pfa_cusum(i), Pd_cusum(i), sprintf('%.2f', tau_values(i)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
        'FontSize', 8, 'Color', 'blue');
end

% Markers for chi-squared thresholds
for i = 1:2:length(chi_thresholds)
    text(Pfa_chi(i), Pd_chi(i), sprintf('%.2f', chi_thresholds(i)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
        'FontSize', 8, 'Color', 'red');
end

% AUC for both detectors
auc_cusum = trapz(Pfa_cusum, Pd_cusum); % AUC for CUSUM
auc_chi = trapz(Pfa_chi, Pd_chi); % AUC for Chi-Squared
annotation('textbox', [0.7, 0.05, 0.25, 0.1], 'String', sprintf('CUSUM AUC: %.4f\nChi-Squared AUC: %.4f', auc_cusum, auc_chi), ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'black');
hold off;

% Function to run a single simulation with both CUSUM and Chi-Squared detectors
function [alarm_counts_cusum, alarm_counts_chi] = run_simulation(tau_values, chi_thresholds, with_attack, ...
    num_timesteps, attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L, Sigma)

    x = zeros(2, num_timesteps);
    x_hat = zeros(2, num_timesteps);
    u = zeros(1, num_timesteps);
    r = zeros(1, num_timesteps);
    cusum = zeros(size(tau_values));
    alarm_counts_cusum = zeros(size(tau_values));
    alarm_counts_chi = zeros(size(chi_thresholds));

    % Initialize state
    x(:, 1) = randn(2, 1);
    x_hat(:, 1) = x(:, 1);

    % Simulation loop
    for k = 2:num_timesteps
        % State update
        v = mvnrnd([0; 0], R1)';
        x(:, k) = F*x(:, k-1) + G*u(k-1) + v;

        % Measurement
        n = sqrt(R2) * randn();
        y = C*x(:, k) + n;

        % Add attack if in attack window and with_attack is true
        if with_attack && k >= attack_start
            y = y + attack_magnitude;
        end

        % State estimate
        x_hat(:, k) = F*x_hat(:, k-1) + G*u(k-1);

        % Residual
        r_k = y - C*x_hat(:, k);

        % Update state estimate
        x_hat(:, k) = x_hat(:, k) + L*r_k;

        % Control input for next step
        u(k) = K*x_hat(:, k);

        % Quadratic Residual (CUSUM)
%         z_k = r_k' * inv(Sigma) * r_k;
        z_k = abs(r_k);
        cusum = max(0, cusum + z_k - bias);

        % Chi-Squared detector
        chi_sq_stat = r_k' * inv(Sigma) * r_k;

        % Check for CUSUM alarms for each tau value
        for tau_idx = 1:length(tau_values)
            if cusum(tau_idx) > tau_values(tau_idx)
                if ~with_attack || (with_attack && k >= attack_start)
                    alarm_counts_cusum(tau_idx) = 1;
                end
            end
        end

        % Check for Chi-Squared alarms for each threshold
        for chi_idx = 1:length(chi_thresholds)
            if chi_sq_stat > chi_thresholds(chi_idx)
                if ~with_attack || (with_attack && k >= attack_start)
                    alarm_counts_chi(chi_idx) = 1;
                end
            end
        end
    end
end