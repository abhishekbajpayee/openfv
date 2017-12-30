function [r1new,iter,max_err_r1] = NR_1eq(r1,r2,z1,z2,n1,n2,tol)

r1new      = r1;
iter       = 0;
max_err_r1 = 1000;

while max_err_r1 > tol

    r1_old     = r1new;
    [f,df_r1]  = f_eval_1eq(r1_old,r2,z1,z2,n1,n2);
    r1new      = r1_old - f./df_r1;
    err_r1     = abs(r1new-r1_old);
    iter       = iter + 1;
    max_err_r1 = max(err_r1);
    
end


end

