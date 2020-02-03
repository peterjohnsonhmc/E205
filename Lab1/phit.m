function p = phit(ztk, sigma_hit, ztk_star, zmax) %dont take in xt or m,
%PHIT Find the probability of a good measurement 
%   Correct range with local measurement noise measured as a gaussian
%   Instead of ray casting with a map, we take in a known true measurement
    if (0 <= ztk && ztk <= zmax) %In bounds
        N = normpdf(ztk, ztk_star, sigma_hit); % eq(6.4)
        eta = normcdf([0 zmax], ztk_star, sigma_hit); % eq (6.6)
        eta = (eta(2)-eta(1))^-1; %normcdf is integral over just the bounds
        p = N*eta;
    else
        p = 0;
    end
end
