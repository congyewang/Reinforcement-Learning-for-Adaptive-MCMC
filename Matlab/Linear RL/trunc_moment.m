% computes \int_L^U (y-x)^2 N(y;m,s^2) dy

function out = trunc_moment(L,U,x,m,s)

l = (L - m) / s;
u = (U - m) / s;
Nl = my_normpdf(l,0,1); % using custom functions to allow symbolic differentiation
Nu = my_normpdf(u,0,1);
Cl = my_normcdf(l);
Cu = my_normcdf(u);

if ~(isinf(L)||isinf(U))
    out = (m - x)^2 * (Cu - Cl) + 2 * (m - x) * s * (Nl - Nu) ...
          + s^2 * ( Cu - Cl - (u*Nu - l*Nl) );     
elseif isinf(U)
    out = (m - x)^2 * (1 - Cl) + 2 * (m - x) * s * Nl ...
          + s^2 * ( 1 - Cl + l*Nl );
elseif isinf(-L)
    out = (m - x)^2 * Cu - 2 * (m - x) * s * Nu ...
          + s^2 * ( Cu - u*Nu );
end

end