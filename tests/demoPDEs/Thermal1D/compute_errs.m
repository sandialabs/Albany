close all; 
clear all; 

%define parameters 
kappa = 1.6; %thermal conductivity
c = 2.09e3; %heat capacity
rho = 1.0; %density 
k = sqrt(kappa/c/rho);
w = 0.0015; %parameter
a = 250;
dt = 1.0e3; %time step
tf = 1.0e5; %final time 
t = [0:dt:tf]; 
dx = 1/20; 
x = [0:dx:1]; 
T = ncread('thermal1D_out.exo', 'vals_nod_var2'); 
%size(T)
Texact = 0*T; 
for i=1:length(x)
  Texact(i,:) = a*cosh(w/k*x(i))*exp(w*w*t); 
end  
%size(Texact) 

K = length(t); 
fig1=figure(1);
winsize = get(fig1,'Position');
winsize(1:2) = [0 0];
Movie=moviein(K,fig1,winsize);
set(fig1,'NextPlot','replacechildren')

rel_err = []; 
 
for i=1:K 
  set(gca,'NextPlot','replacechildren') ; 
  plot(x, T(:,i), 'b'); 
  hold on; 
  plot(x, Texact(:,i), '--r'); 
  err = norm(T(:,i)-Texact(:,i))/norm(Texact(:,i)); 
  rel_err = [rel_err; err]; 
  xlabel('x'); 
  ylabel('Temp'); 
  legend('Computed','Exact', 'Location','NorthWest'); 
  title(['Temperature solution at t = ', num2str(t(i))]); 
  axis([0 1 250 350])  
  pause(0.1)
  Movie(:,i)=getframe(fig1,winsize);
  mov(i) = getframe(gcf);
end 
movie2avi(Movie,'Thermal_1D_Verification_MMS.avi','fps',2,'quality',10,'Compression','None');
X = ['Average relative error over time = ', num2str(mean(rel_err))]; 
disp(X) 
disp([' ']); 
figure(); 
plot(t, rel_err); 
xlabel('time'); 
ylabel('Relative Error'); 
title('Relative Errors w.r.t. Exact Solution');  


