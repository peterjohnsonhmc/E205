function p = pshort(ztk,lambda_short, ztk_star) %zmax does not matter
%PHSORT FInd probability of an unexpectedly close object
%   Exponential Distribution
    
    if (0 <= ztk && ztk <= ztk_star) %In bounds
        mu = 1/lambda_short;
        N = exppdf(ztk, mu); % eq(6.7)
        eta = expcdf([0 ztk_star],mu); % eq (6.9)
        eta = (eta(2)-eta(1))^-1; 
        %Closed form eta from eq 6.9
        %eta_alt = 1/(1-exp(-lambda_short*ztk_star)); 
        p = N*eta;
    else
        p = 0;
    end
end

