function p = prand(ztk, zmax) 
%PRAND Find probability of a random measurment
%   Uniform distribution between 0 and zmax
    if (0 <= ztk && ztk <= zmax) % eq (6.11)
        p = 1/zmax;
    else
        p = 0;
    end
end