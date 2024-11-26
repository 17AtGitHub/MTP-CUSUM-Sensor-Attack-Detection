%% System Matrices (unchanged)
A = [0.96, 0.99, -0.88, 0.56;
     0, 0.98, 0.75, -0.65;
     0, 0, 0.97, 0.95;
     0, 0, 0, 0.94];
C = [1, 0, 0, 0;
     0, 0, 0, 0;
     0, 0, 1, 0;
     0, 0, 0, 0];

%% Simulation Parameters
T = 1000; % Total timesteps
change_point = 500; % Timestep at which the variance change is introduced

% Define noise covariances
initial_covariance = 0.1 * eye(4); % Covariance before the change point
changed_covariance = 0.5 * eye(4); % Covariance after the change point

% Initial state
Z_no_change = zeros(4, T);
Y_no_change = zeros(4, T);
Z_change = zeros(4, T);
Y_change = zeros(4, T);

% Initial conditions
Z_no_change(:, 1) = [1; 1; 1; 1];
Z_change(:, 1) = [1; 1; 1; 1];

%% Process Simulation with Variance Change at Change Point
for i = 2:T
    % Generate noise with initial covariance before the change point
    if i >= change_point
        W_variance_changed = mvnrnd(zeros(4,1), changed_covariance)';
    else
        W_variance_changed = mvnrnd(zeros(4,1), initial_covariance)';
    end
    
    % Update state and observation for both scenarios
    Z_no_change(:, i) = A * Z_no_change(:, i-1) + mvnrnd(zeros(4,1), initial_covariance)'; % No change scenario
    Y_no_change(:, i) = C * Z_no_change(:, i) + mvnrnd(zeros(4,1), initial_covariance)';
    
    Z_change(:, i) = A * Z_change(:, i-1) + W_variance_changed; % With variance change
    Y_change(:, i) = C * Z_change(:, i) + W_variance_changed;
end

%% Plot Results
figure;

% Plot observed process without variance change
subplot(2, 1, 1);
plot(1:T, Y_no_change');
title('Observed Process Y without Variance Change');
xlabel('Time Step');
ylabel('Observation Value');
legend('Y_1', 'Y_2', 'Y_3', 'Y_4');
grid on;

% Plot observed process with variance change
subplot(2, 1, 2);
plot(1:T, Y_change');
title('Observed Process Y with Variance Change Introduced at t = 500');
xlabel('Time Step');
ylabel('Observation Value');
legend('Y_1', 'Y_2', 'Y_3', 'Y_4');
grid on;
