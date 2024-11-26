% Simulation parameters
num_simulations = 100;
num_timesteps = 100;

% System matrices
F = [0.84 0.23; -0.47 0.12];
G = [0.07; 0.23];
C = [1 0];
K = [-1.85 -0.96];
R1 = [0.45 -0.11; -0.11 0.20];
R2 = 1;
L = [0.31; -0.21];

attack_start = 80;
attack_magnitude = 3;
bias = 1.5;  % Fixed bias
desired_false_alarm_rate = 0.05;  % Desired false alarm rate
N = 50;  % Number of states in Markov chain
sigma_i = sqrt(R2);  % Standard deviation of residuals
tolerance = 1e-5;  % Tolerance for bisection method

% Find the optimal threshold using the bisection method
tau_opt = find_optimal_tau(bias, desired_false_alarm_rate, N, sigma_i, tolerance);

% Run your simulation with the obtained optimal tau
false_alarms = 0;
detections = 0;

% Simulations without attacks (for false alarms)
for sim = 1:num_simulations
    [false_alarm, ~] = run_simulation(tau_opt, false, num_timesteps, ...
        attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L);
    false_alarms = false_alarms + false_alarm;
end

% Simulations with attacks (for detections)
for sim = 1:num_simulations
    [~, detection] = run_simulation(tau_opt, true, num_timesteps, ...
        attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L);
    detections = detections + detection;
end

% Getting probabilities
Pfa = false_alarms / num_simulations;
Pd = detections / num_simulations;

% % ROC curve
% figure;
% plot(Pfa, Pd, 'b-o', 'LineWidth', 1.5);
% xlabel('Probability of False Alarm (Pfa)');
% ylabel('Probability of Detection (Pd)');
% title('ROC Curve for Attack Detection Model');
% grid on;
% 
% % Markers for tau values
% for i = 1:2:length(tau_values) 
%     text(Pfa(i), Pd(i), sprintf('%.2f', tau_values(i)), ...
%         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', ...
%         'FontSize', 8, 'Color', 'red');
% end
% 
% % Calculating AUC (Area Under the Curve)
% auc = trapz(Pfa, Pd); % Integrate using trapezoidal method
% annotation('textbox', [0.7, 0.05, 0.25, 0.1], 'String', sprintf('AUC: %.4f', auc), ...
%            'FitBoxToText', 'on', 'BackgroundColor', 'white', 'EdgeColor', 'black');
% 
% hold off;
