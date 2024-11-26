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
T = 10000; % Total timesteps
Z = zeros(4, T); % State process
Y = zeros(4, T); % Observation process
sigma = 1; % Kernel parameter

% Noise
W = mvnrnd(zeros(4,1), eye(4), T)'; % Actuation noise W ~ N(0, I)
V = mvnrnd(zeros(4,1), eye(4), T)'; % Observation noise V ~ N(0, I)

% Initial state
Z(:, 1) = [1; 1; 1; 1]; 

% Process Simulation
for i = 2:T
    Z(:, i) = A * Z(:, i-1) + W(:, i);
    Y(:, i) = C * Z(:, i) + V(:, i);
end

% Reference Dataset
D_h = Y(:, 1:100);

%% CUSUM MMD Test Parameters
r = 50; % Buffer size
thresholds = 0.01:0.02:0.5; % Varying thresholds for the test
Delta = 0.006; % Bias delta

ARL_vals = zeros(length(thresholds), 1);

for idx = 1:length(thresholds)
    b = thresholds(idx); % Threshold for this run
    alarms = [];
    s_t = 0;
    s_min = 0;

    for t = 1:floor(T/r)
        % Update buffer with new window
        B_r = Y(:, (t-1)*r + 1:t*r);
        
        % Compute MMD value
        MMD_value = compute_MMD(B_r, D_h, sigma);
        
        % Update the test statistic
        s_t = s_t + MMD_value - Delta;
        s_min = min(s_t, s_min);
        
        % Check for false alarm
        if (s_t - s_min) > b
            alarms = [alarms, t * r];
            s_t = 0; % Reset the statistic after alarm
            s_min = 0;
        end
    end
    
    % Calculate ARL and ADD for this threshold
    if ~isempty(alarms)
        ARL_vals(idx) = mean(diff([0, alarms]));
    else
        ARL_vals(idx) = T; % No alarms, treat as max run length
    end
end

%% Plotting Results
figure;
subplot(1,1,1);
plot(thresholds, log(ARL_vals), '-o'); % Add '-o' for line with markers
title('Log(ARL) vs Threshold');
xlabel('Threshold');
ylabel('Log(ARL)');