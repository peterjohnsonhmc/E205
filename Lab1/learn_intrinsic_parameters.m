function Theta = learn_intrinsic_parameters(zt, zt_star, sigma_hit, lambda_short, zmax_range) 
%LEARN_INTRINSIC_PARAMETERS Find distribution weights and parameters
%   Typically uses ray casting, but we will pass in a single known zt_star
    for i=1:15 %typically enough for convergence to be reached
        ehit = ones(size(zt));  %Just initialize first
        eshort = ones(size(zt));
        emax = ones(size(zt));
        erand = ones(size(zt));
        for k=1:length(zt)
            eta = phit(zt(k), sigma_hit, zt_star, zmax_range)+ pshort(zt(k),lambda_short, zt_star) + pmax(zt(k), zmax_range) + prand(zt(k), zmax_range);
            %Would calculate zt_star here
            ehit(k) = eta*phit(zt(k), sigma_hit, zt_star, zmax_range);
            eshort(k) = eta*pshort(zt(k),lambda_short, zt_star);
            emax(k) = eta*pmax(zt(k), zmax_range);
            erand(k) = eta*prand(zt(k), zmax_range);
        end
        Z = sum(zt);
        zhit = Z^(-1)*sum(ehit);
        zshort = Z^(-1)*sum(eshort);
        zmax = Z^(-1)*sum(emax);
        zrand = Z^(-1)*sum(erand);
        sigma_hit = sqrt(1/sum(ehit)*sum(ehit.*(zt-zt_star).^2));
        lambda_short = sum(eshort)/sum(eshort.*zt);
        Theta = [zhit, zshort,zmax, zrand,sigma_hit, lambda_short]
    end  
        
        
end