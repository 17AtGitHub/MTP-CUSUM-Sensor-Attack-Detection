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

% Generate actuation and observation noise
W = mvnrnd(zeros(4,1), eye(4), T)'; % Actuation noise W ~ N(0, I)
V = mvnrnd(zeros(4,1), eye(4), T)'; % Observation noise V ~ N(0, I)

% Initialize storage for state and observation processes
Z_no_change = zeros(4, T);
Y_no_change = zeros(4, T);
Z_change = zeros(4, T);
Y_change = zeros(4, T);

% Initial conditions
Z_no_change(:, 1) = [1; 1; 1; 1];
Z_change(:, 1) = [1; 1; 1; 1];

%% Process Simulation with Variance Change at Change Point
for i = 2:T
    if i >= change_point
        % After the change point, scale the pre-generated noise for reduced variance
        W_variance_changed = sqrt(0.5) * W(:, i); % Scale noise to have covariance 0.05 * I
    else
        % Use the original pre-generated noise up to the change point
        W_variance_changed = W(:, i);
    end
    
    % Update for both scenarios
    Z_no_change(:, i) = A * Z_no_change(:, i-1) + W(:, i); % No variance change scenario
    Y_no_change(:, i) = C * Z_no_change(:, i) + V(:, i);
    
    Z_change(:, i) = A * Z_change(:, i-1) + W_variance_changed; % With variance change scenario
    Y_change(:, i) = C * Z_change(:, i) + V(:, i);
end

%% Plot Results
figure;

% Plot observed process without variance change
subplot(2, 1, 1);
plot(1:T, Y_no_change');
title('Observed Process Y without Mean SHift');
xlabel('Time Step');
ylabel('Observation Value');
legend('Y_1', 'Y_2', 'Y_3', 'Y_4');
grid on;

% Plot observed process with variance change
subplot(2, 1, 2);
plot(1:T, Y_change');
title('Observed Process Y with Mean Shift Introduced at t = 500');
xlabel('Time Step');
ylabel('Observation Value');
legend('Y_1', 'Y_2', 'Y_3', 'Y_4');
grid on;
