function p = pmax(ztk, zmax) 
%PMAX Find probability of a max distance value/ failed reading
%   Point or a uniform distribution
    if (ztk == zmax) % eq (6.10)
        p = 1;
    else
        p = 0;
    end
end