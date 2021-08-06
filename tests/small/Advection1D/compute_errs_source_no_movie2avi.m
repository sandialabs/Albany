close all; 
clear all; 

%define parameters 
a = 10.0;
u = ncread('advection1D_dx50_dt1em5_out.exo', 'vals_nod_var1'); 
t = ncread('advection1D_dx50_dt1em5_out.exo', 'time_whole'); 
x = ncread('advection1D_dx50_dt1em5_out.exo', 'coordx'); 

uexact = 0*u;
for i=1:length(x)
  uexact(i,:) = sin(x(i) - a*t);  
end  
K = length(t); 

%fig1=figure(1);
%winsize = get(fig1,'Position');
%winsize(1:2) = [0 0];
%Movie=moviein(K,fig1,winsize);
%set(fig1,'NextPlot','replacechildren')

rel_err = []; 
soln_avg_rel_err = [];  
for i=1:K
  %set(gca,'NextPlot','replacechildren') ; 
  %plot(x, T(:,i), 'b');
  soln_avg_computed = mean(u(:,i));  
  %hold on; 
  %plot(x, Texact(:,i), '--r'); 
  soln_avg_exact = mean(uexact(:,i));
   err = norm(u(:,i)-uexact(:,i))/norm(uexact(:,i)); 

  rel_err = [rel_err; err]; 
  soln_avg_rel_err = [soln_avg_rel_err; norm(soln_avg_computed-soln_avg_exact)/norm(soln_avg_exact)]; 
  %xlabel('x'); 
  %ylabel('Solution'); 
  %legend('Computed','Exact', 'Location','NorthWest'); 
  %title(['solution at t = ', num2str(t(i))]); 
  %axis([0 1 0 4])  
  %pause(0.1)
  %Movie(:,i)=getframe(fig1,winsize);
  %mov(i) = getframe(gcf);
  %n = strcat("advection_", num2str(i)); 
  %name = strcat(n, ".png") 
  %saveas(gcf,name)
end 

%To convert the pngs to mp4, run:
%  ffmpeg -framerate 10 -start_number 1 -i thermal_%1d.png -r 5 -vf scale=-2:1080,setsar=1 -pix_fmt yuv420p thermal_MMS.mp4

X = ['Average solution relative error over time = ', num2str(mean(rel_err))]; 
disp(X) 
disp([' ']); 
X = ['Solution relative error at last time = ', num2str(err)]; 
disp(X) 
disp([' ']); 
figure(); 
plot(t, rel_err); 
xlabel('time'); 
ylabel('Relative Error'); 
title('Relative Errors w.r.t. Exact Solution');  

