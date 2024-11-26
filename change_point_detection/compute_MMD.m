% MMD computation function
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
