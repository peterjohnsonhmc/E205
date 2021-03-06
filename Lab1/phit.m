function p = phit(ztk, sigma_hit, ztk_star, zmax) %dont take in xt or m,
%PHIT Find the probability of a good measurement 
%   Correct range with local measurement noise measured as a gaussian
%   Instead of ray casting with a map, we take in a known true measurement
    if (0 <= ztk && ztk <= zmax)                      %Measurment is in bounds
        N = normpdf(ztk, ztk_star, sigma_hit);        % eq(6.4) pdf is a relative probability
        eta = normcdf([0 zmax], ztk_star, sigma_hit); % eq (6.6) 
        eta = (eta(2)-eta(1))^-1;                     %normcdf is integral of pdf over the bounds
        p = N*eta;                                    %cdf is used to normalize probability
        %When probabilities are small, want to ensure they are nonzero
        if p == 0                                     
            p=0.0001;
        end
    else
        p = 0;
    end
end
