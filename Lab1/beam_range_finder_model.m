function q = beam_range_finder_model(zt,zt_star, zmax_range, Theta ) 
%BEAM_RANGE_FINDER_MODEL alogrithm for likelihood of range scan
%   Does not implement ray casting to find different zt_star
%   All measurements must be for a single distance
    % Set mixing weights and parameters
    zhit   = Theta(1);
    zshort = Theta(2);
    zmax   = Theta(3);
    zrand  = Theta(4);
    sigma_hit = Theta(5);
    lambda_short = Theta(6);
    q=1;
    for k=1:length(zt)
        p=zhit*phit(zt(k), sigma_hit, zt_star, zmax_range)+zshort * pshort(zt(k),lambda_short, zt_star)+zmax*pmax(zt(k), zmax_range)+zrand*prand(zt(k), zmax);
        q=q*p;
    end
end