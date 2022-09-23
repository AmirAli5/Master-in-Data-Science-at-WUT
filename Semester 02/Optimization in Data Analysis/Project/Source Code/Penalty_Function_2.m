clear all; 
close all;  
clc;

%% Parameterisation

% Rate of conv % ? > 1      % Shift   % Weight       % Constraint   
alfa = 0.1;   beta = 4;     c = 1;    t = 1;         xn = 10; 
% Gamma's                   % Treshold (e1/e2)       % Fit initial
Gama1 = 0; Gama2 = 1;       e1 = 0.1; e2 = 0.1;      Jbest = 0;

%%  Subtitution
a = [20 30 4 5 5 5 5 5 5 5];          
v1= a(1); v2 = a(2); v3 = a(3); 
v4= a(4:10); v = [v1 v2 v3 v4];

%% Initialisation 
u = [5 5 5 5 5 5 5];                     % Initial Solution
x = [20 0 0 50 0 0 0];                  % State trajectory
it = 0;

%% Calculation and computation
while abs(Gama1-Gama2) > e1                 % First Condition check
    Gama1 = Gama2;
    it = it+1;
    					% Performance index J fitness function
    Fit = @(u)   (sum(x.^2+ 3*u.^2)...      
               + (t*((((x(1) + sum(u))-v(1))^ 2)...
               + sum((u-v(4:10)).* max(0, u-v(4:10))...
               + (-u-v(4:10)).* max(0, -u-v(4:10)))...
               + ((u(4)-v(3))^2) + (((x(1) + sum(u(1:3))-v(2))^2)))));

    [u,Jx] = fminsearch(Fit,u);             % Local minimum search of J

    for j = 1:7                             
         if j < 7                           % Computing x values up to x6
            x(j+1) = x(j)+u(j);
         end
         if  j == 7
             xn = x(j)+u(j);                % Assigning given value of x7
         end
    end

    for j = 1:7
        b =  u(j)-5;  c = -u(j)-5;
        r4(j) = max([0 (u(j)-5) (-u(j)-5)]);    % Condition of Ui <= 5 
    end
    r = [xn-a(1),x(4)-a(2),u(4)-a(3),r4];   % Next violation values

    for j = 1:7
        a4(j) = v(j+3)+r4(j);               
    end
    ai= [v(1)+r(1),v(2)+r(2),v(3)+r(3),a4]; % Next matrix of constraints 

    if Gama2 > e1 && Gama2 < c              % Stopping critiria check
        for j = 1:10
            v(j) = a(j)-r(j);               % Update of v matrix
        end
            c = alfa * Gama2;               % Update of the penalty shift
    end

    if Gama2 > e1 && Gama2 >= c             % Stopping critiria check
        t = beta*t;                         % Weight coefficient
        for j = 1:10
            v(j) = a(j)-(1/beta)*r(j);      % Update of v matrix        
        end
    end

    if abs(Jbest-Jx) < e1 || Gama2 < e1     % Stopping criteria to satisfy
      break                                 % constraints.
    else
        Jbest = Jx;                         % Update of Performance index
    end

    Gama2 = norm(ai-a);     % First stopping criteria's check (While loop)
    [x;u]'
    Jx
end 
