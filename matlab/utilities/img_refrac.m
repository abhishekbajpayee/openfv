function [XB,XD,max_err_rB] = img_refrac(XC,XP,Zw,t,n)
% [XB,rB,max_err_rB] = img_refrac(XC,XP,Zw,t,n)
% This function iteratively solves refraction imaging model
%
% INPUTS:
% XC       - 3 x 1 vector containing the coordinates of the camera's center of projection (COP)
% XP       - 3 x N vector containing the coordinates of each point
% na,ng,nw - index of refraction of air, glass and water
%
% OUTPUTS:
% XB       - 4 x N vector containing the coordinates where the ray from each point intersects the air-facing side of the interface (wall)
% rB       - radial distance of points in XB from the Z axis
% iter     - number of iterations for the Newton-Raphson solver

if any(XC(3,:) > Zw)
    msg = 'Z-coordinate of camera center cannot be larger than Z-coordinate of wall';
    warning('%s',msg);
end

%% Initialize

[r,N] = size(XP);
Zcam = XC(3,:);
z1   = (Zw-Zcam).*ones(1,N);
z2   = t.*ones(1,N);
z3   = (XP(3,:)-Zcam) - (z2+z1);

n1 = n(1);
n2 = n(2);
n3 = n(3);

XB      = zeros(size(XP));
XB(3,:) = Zw;
XB(4,:) = 1.0;

XD      = zeros(size(XP));
XD(3,:) = Zw+t;
XD(4,:) = 1.0;

if any(z3 < 0)
    msg = 'Z-coordinate of points cannot be less than distance of camera to wall';
    error('%s',msg);
end
if t < 0
    msg = 'Wall thickness cannot be negative';
    error('%s',msg);
end

rPorig = sqrt((XP(1,:)-XC(1,:)).^2 + (XP(2,:)-XC(2,:)).^2);
rP  = rPorig;
rB0 = z1.*rP./(XP(3,:)-Zcam);
rD0 = (z1+z2).*rP./(XP(3,:)-Zcam);

fcheck = zeros(1,N);
gcheck = zeros(1,N);

tol = sqrt(eps);
fg_tol = 0.001;
maxiter = 500;
bi_tol     = 1e-3;
bi_maxiter = 1000;
z3_tol     = 1e-3;

%%

if t == 0;

    rB      = rP;
    i1      = find(z3 == 0);
    i2      = find(z3 ~= 0);

    % Perform Newton-Raphson iteration + check result
    [rB(i2),iter,max_err_rB(i2)] = NR_1eq(rB0(i2),rP(i2),z1(i2),z3(i2),n1,n3,tol);
    
    if any(isnan(rB))
        rdummy                = zeros(1,length(i1));
        [rB(i2),~,err_r2f] = bisection(rB0(i2),rP(i2),rdummy,rP(i2),z1(i2),z3(i2),n1,n3,bi_tol);
    end
    
    [fcheck(i2),df_dr1]      = f_eval_1eq(rB(i2),rP(i2),z1(i2),z3(i2),n1,n3);
    if any(isnan(fcheck))
        msg = 'Warning: f has a NaN';
        warning('%s',msg);
    end
    if max(abs(fcheck)) > tol
        msg = ['Warning: max values of f = ' num2str(max(abs(fcheck))) '. This may be larger than it should be'];
        warning('%s',msg);
    end
    

else    %%% t ~= 0
    
    rB   = rP;
    rD   = rP;
    i1      = find(z3 < z3_tol);
    i2      = find(z3 >= z3_tol);

    % First find rB for points on the back side of wall
    if ~isempty(i1)
        rdummy = zeros(1,length(i1));
        [rB(i1),iter,err_r2f] = bisection(rB0(i1),rD0(i1),rdummy,rP(i1),z1(i1),z2(i1),n1,n2,bi_tol);
        [fcheck(i1),df_dr1]   = f_eval_1eq(rB(i1),rP(i1),z1(i1),z2(i1),n1,n2);
    end
    
    % Perform Newton-Raphson iteration + check result
    [rB(i2),rD(i2),iter,max_err_rB(i2),max_err_rD] = ...
        NR_2eq(rB0(i2),rD0(i2),rP(i2),z1(i2),z2(i2),z3(i2),n1,n2,n3,tol,maxiter);
    
    % If N-R doesn't converge => use bisection
    if any(isnan(rB)) | any(rB == inf)
        
        % First find rB for points on the back side of wall
        i1      = find(z3 < z3_tol);
        if ~isempty(i1)
            rdummy = zeros(1,length(i1));
            [rB(i1),iter,err_r2f] = bisection(rB0(i1),rD0(i1),rdummy,rP(i1),z1(i1),z2(i1),n1,n2,bi_tol);
            [fcheck(i1),df_dr1]   = f_eval_1eq(rB(i1),rP(i1),z1(i1),z2(i1),n1,n2);
        end
        
        nan_ind    = find(isnan(rB));
        [rB(nan_ind),rD(nan_ind),~] = ...
            refrac_solve_bisec(rB0(nan_ind),rD0(nan_ind),rP(nan_ind),z1(nan_ind),z2(nan_ind),z3(nan_ind),n1,n2,n3,bi_tol,bi_maxiter);
    end
    
    [fcheck(i2),df1,df2,gcheck(i2),dg1,dg2]          = f_eval_2eq(rB(i2),rD(i2),rP(i2),z1(i2),z2(i2),z3(i2),n1,n2,n3);
    
    if max(abs(fcheck)) > fg_tol | max(abs(gcheck)) > fg_tol
        msg = ['Warning: max values of f = ' num2str(max(abs(fcheck))) ', max values of g = ' num2str(max(abs(gcheck))) '. These may be larger than they should be'];
        warning('%s',msg);
    end

    if any(isnan(fcheck)) | any(isnan(gcheck))
        msg = 'Warning: f or g has a NaN';
        warning('%s',msg);
    end
    

end

% 2nd & 3rd equation from refraction model (constraint on ray direction)
phi     = atan2((XP(2,:)-XC(2,:)),(XP(1,:)-XC(1,:)));
XB(1,:) = rB.*cos(phi) + XC(1,:);
XB(2,:) = rB.*sin(phi) + XC(2,:);

XD(1,:) = rD.*cos(phi) + XC(1,:);
XD(2,:) = rD.*sin(phi) + XC(2,:);

end





