function [r1new,r2new,iter,max_err_r1,max_err_r2] = NR_2eq(r1,r2,r3,z1,z2,z3,n1,n2,n3,tol,maxiter)

r1new      = r1;
r2new      = r2;
iter       = 0;
max_err_r1 = 1000;
max_err_r2 = max_err_r1;
while max_err_r1 > tol | max_err_r2 > tol & iter < maxiter

    r1_old = r1new;
    r2_old = r2new;

    [f,df_dr1,df_dr2,g,dg_dr1,dg_dr2] = f_eval_2eq(r1_old,r2_old,r3,z1,z2,z3,n1,n2,n3);

    denom = (df_dr1.*dg_dr2 - df_dr2.*dg_dr1);
    r1new = r1_old - (f.*dg_dr2 - g.*df_dr2)./denom;
    r2new = r2_old - (g.*df_dr1 - f.*dg_dr1)./denom;

    iter   = iter + 1;
    err_r1 = abs(r1new - r1_old);
    err_r2 = abs(r2new - r2_old);

    max_err_r1 = max(err_r1);
    max_err_r2 = max(err_r2);

end

end

