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

Deltas = [0.05, 0.08, 0.1]; % Range of Delta values to test
thresholds = 0.01:0.05:0.5; % Thresholds to test
num_repeats = 5; % Number of repetitions for averaging ARL

% Define colors and markers for each Delta value
colors = ['r', 'g', 'b']; % Colors for plotting
markers = ['o', 's', '^']; % Different markers for each Delta

figure;
hold on;

for delta_idx = 1:length(Deltas)
    Delta = Deltas(delta_idx);
    fprintf('Simulating for Delta = %f\n', Delta);
    arl_results = zeros(length(thresholds), num_repeats); % Store ARL results

    for idx = 1:length(thresholds)
        threshold = thresholds(idx);
        for repeat = 1:num_repeats
            % Reinitialize state and observation processes for each repeat
            Z = zeros(4, T); % State process
            Y = zeros(4, T); % Observation process

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
            D_h = Y(:, 1:200);

            % Initialize CUSUM test variables
            alarms = [];
            s_t = 0;
            s_min = 0;

            % Initialize vectors for MMD, residuals, and cumulative sum
            mmd = zeros(1, floor(T/r) - 3); % Adjusted for skipping first 3 windows
            residuals = zeros(1, floor(T/r) - 3); % Store residuals
            cumulative_sum = zeros(1, floor(T/r) - 3); % Store cumulative sum

            for t = 1:floor(T/r)
                % Update buffer with new window
                B_r = Y(:, (t-1)*r + 1:t*r);

                % Compute MMD value
                MMD_value = compute_MMD(B_r, D_h, sigma);

                % Calculate residual
                residual = MMD_value - Delta;

                % Update the test statistic
                s_t = s_t + residual;
                s_min = min(s_t, s_min);

                % If past initial 3 windows, store values
                if t > 3
                    mmd(t-3) = MMD_value;
                    residuals(t-3) = residual;
                    cumulative_sum(t-3) = s_t;
                end

                % Check for false alarm
                if (s_t - s_min) > threshold
                    alarms = [alarms, t * r];
                    s_t = 0; % Reset the statistic after alarm
                    s_min = 0;
                end
            end

            % Calculate run lengths
            if ~isempty(alarms)
                run_lengths = diff([0, alarms]);
                if ~isempty(run_lengths) && length(run_lengths) > 1
                    run_lengths = run_lengths(2:end); % Ignore the first run length
                end
            else
                run_lengths = 1; % If no alarms, set a small value representing no alarms occurred
            end

            % Store average run length for this repeat
            arl_results(idx, repeat) = mean(run_lengths);
        end

        fprintf('ARL for threshold %f = %f\n', threshold, mean(arl_results(idx, :)));
    end

    % Average ARL across repetitions
    avg_arl = mean(arl_results, 2);
    % Plot ARL vs Threshold with distinct markers
    plot(thresholds, log10(avg_arl), ['-' markers(delta_idx)], ...
         'Color', colors(delta_idx), ...
         'DisplayName', sprintf('\\Delta = %.2f', Delta), ...
         'MarkerSize', 8, ...
         'LineWidth', 1.5);
end

hold off;
title('Log(Average ARL) vs Threshold for Different \Delta Values');
xlabel('Threshold');
ylabel('Log_{10}(Average ARL)');
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