close all; 
clear all; 

%define parameters 
kappa = 1.6; %thermal conductivity
c = 2.09e3; %heat capacity
rho = 1.0; %density 
k = sqrt(kappa/c/rho);
w = 0.0015; %parameter
a = 250;
dt = 10.0; %time step
tf = 1e5; %final time 
t = [0:dt:tf]; 
dx = 1/20; 
x = [0:dx:1]; 
for i=1:length(x)
  Texact(i,:) = a*cosh(w/k*x(i))*exp(w*w*t); 
end  
size(Texact)  
for i=1:length(t) 
  set(gca,'NextPlot','replacechildren') ; 
  plot(x, Texact(:,i), '--r'); 
  xlabel('x'); 
  ylabel('Temp'); 
  title(['Exact temperature solution at t = ', num2str(t(i))]);
  axis([0 1 250 260])  
  pause(0.1) 
end 
