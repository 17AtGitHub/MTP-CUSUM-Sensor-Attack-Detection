function [false_alarm, detection] = run_simulation(tau, with_attack, num_timesteps, attack_start, attack_magnitude, bias, F, G, C, K, R1, R2, L)
    x = zeros(2, num_timesteps);
    x_hat = zeros(2, num_timesteps);
    u = zeros(1, num_timesteps);
    r = zeros(1, num_timesteps);
    cusum = 0;
    false_alarm = 0;
    detection = 0;
    
    % State Initialization
    x(:, 1) = randn(2, 1);
    x_hat(:, 1) = x(:, 1);
    
    % Simulation loop
    for k = 2:num_timesteps
        % State update
        v = mvnrnd([0; 0], R1)';
        x(:, k) = F * x(:, k-1) + G * u(k-1) + v;
        
        % Measurement
        n = sqrt(R2) * randn();
        y = C * x(:, k) + n;
        
        % Add attack if in attack window and with_attack is true
        if with_attack && k >= attack_start
            y = y + attack_magnitude;
        end
        
        % State estimate
        x_hat(:, k) = F * x_hat(:, k-1) + G * u(k-1);
        
        % Residual
        r(k) = y - C * x_hat(:, k);
        
        % Update state estimate
        x_hat(:, k) = x_hat(:, k) + L * r(k);
        
        % Control input for next step
        u(k) = K * x_hat(:, k);
        
        % CUSUM
        z_k = abs(r(k));
        cusum = max(0, cusum + z_k - bias);
        
        % Check for alarm
        if cusum > tau
            if ~with_attack
                false_alarm = 1;
                return;
            elseif k >= attack_start
                detection = 1;
                return;
            end
        end
    end
end