figure; hold on;

sum = 0;

for i=2:8
    adotb = (C_ua(i+1,1)-C_ua(i,1))*(C_ua(i-1,1)-C_ua(i,1)) + ...
            (C_ua(i+1,2)-C_ua(i,2))*(C_ua(i-1,2)-C_ua(i,2)) + ...
            (C_ua(i+1,3)-C_ua(i,3))*(C_ua(i-1,3)-C_ua(i,3));
            
    amag = sqrt((C_ua(i-1,1)-C_ua(i,1))^2 + ...
                (C_ua(i-1,2)-C_ua(i,2))^2 + ...
                (C_ua(i-1,3)-C_ua(i,3))^2);
    
    bmag = sqrt((C_ua(i+1,1)-C_ua(i,1))^2 + ...
                (C_ua(i+1,2)-C_ua(i,2))^2 + ...
                (C_ua(i+1,3)-C_ua(i,3))^2);
    
    theta1 = acos(adotb/(amag*bmag));
    scatter(i,theta1,'b+');
    
    adotb = (C_a(i+1,1)-C_a(i,1))*(C_a(i-1,1)-C_a(i,1)) + ...
            (C_a(i+1,2)-C_a(i,2))*(C_a(i-1,2)-C_a(i,2)) + ...
            (C_a(i+1,3)-C_a(i,3))*(C_a(i-1,3)-C_a(i,3));
            
    amag = sqrt((C_a(i-1,1)-C_a(i,1))^2 + ...
                (C_a(i-1,2)-C_a(i,2))^2 + ...
                (C_a(i-1,3)-C_a(i,3))^2);
    
    bmag = sqrt((C_a(i+1,1)-C_a(i,1))^2 + ...
                (C_a(i+1,2)-C_a(i,2))^2 + ...
                (C_a(i+1,3)-C_a(i,3))^2);
    
    theta2 = acos(adotb/(amag*bmag));
    scatter(i,theta2,'r+');
    
    sum = sum + abs(theta1-theta2);
    
end

sum