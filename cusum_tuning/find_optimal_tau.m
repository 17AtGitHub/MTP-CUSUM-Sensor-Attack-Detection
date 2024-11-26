function tau_opt = find_optimal_tau(bias, A_star, N, sigma_i, tol)
    tau_min = 0.1; % Lower bound for tau
    tau_max = 10;  % Upper bound for tau
    
    while abs(tau_max - tau_min) > tol
        tau_mid = (tau_min + tau_max) / 2;
        A_i = compute_false_alarm_rate(tau_mid, bias, N, sigma_i);
        
        if A_i > A_star
            tau_min = tau_mid;  % False alarm rate too high, increase tau
        else
            tau_max = tau_mid;  % False alarm rate too low, decrease tau
        end
    end
    
    tau_opt = (tau_min + tau_max) / 2;
end
