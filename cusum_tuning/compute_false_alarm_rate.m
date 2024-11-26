function A_i = compute_false_alarm_rate(tau, bias, N, sigma_i)
    % Compute the interval size for the Markov chain states
    delta_S = 2 * tau / (2 * N - 1);
    
    % Transition probabilities (approximation using half-normal distribution)
    pr = @(x) normcdf((x + bias) / sigma_i); % CDF of the shifted half-normal
    
    % Construct the transition matrix P
    P = zeros(N+1, N+1);
    for j = 1:N-1
        for k = 1:N
            P(j,k) = pr(k * delta_S - j * delta_S);
        end
    end
    % Fill the last row to make the last state absorbing
    P(N+1, N+1) = 1;

    % Remove the last row and column to get matrix R
    R = P(1:N, 1:N);
    
    % Compute the vector mu
    mu = (eye(N) - R) \ ones(N, 1);
    
    % Calculate the approximate false alarm rate
    A_i = 1 / mu(1);
end
