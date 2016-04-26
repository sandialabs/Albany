
%Note that this function assumes 2 domain.  Can be generalized
%to an arbitrary number of domains. 

%Another assumption here is that displ_x, displ_y and displ_z are
%vals_nod_var1, vals_nod_var2, and vals_nod_var3 in the *exo file,
%respectively.  If they are not, code needs to be modified.

%Input: Schwarz step number, schwarz_no (int) 
function[] = norm_displacements(schwarz_no, step_no) 

file0_exo_name = strcat('cube0_restart_',num2str(step_no),'.exo');
file1_exo_name = strcat('cube1_restart_',num2str(step_no),'.exo');

displ0_current_name = strcat('displ0_current_load', num2str(step_no),'_schwarz',num2str(schwarz_no));
displ1_current_name = strcat('displ1_current_load', num2str(step_no),'_schwarz',num2str(schwarz_no));
error_name = strcat('error_load', num2str(step_no),'_schwarz',num2str(schwarz_no));

disp(['      step_no = ',num2str(step_no)]); 
disp(['      schwarz_no = ',num2str(schwarz_no)]); 

%Here we hard-code 2-norm.  norm_type could be made an input argument.
norm_type = 2; 

%cube0
%x-displacement
displ0_x = ncread(file0_exo_name, 'vals_nod_var1'); 
%y-displacement
displ0_y = ncread(file0_exo_name, 'vals_nod_var2'); 
%z-displacement
displ0_z = ncread(file0_exo_name, 'vals_nod_var3'); 
%get last snapshot
displ0_x = displ0_x(:,end); 
displ0_y = displ0_y(:,end); 
displ0_z = displ0_z(:,end); 
%concatenate into a single displacement vector
displ0 = zeros(3*length(displ0_x),1); 
displ0(1:3:end) = displ0_x; 
displ0(2:3:end) = displ0_y; 
displ0(3:3:end) = displ0_z; 

%cube1
%x-displacement
displ1_x = ncread(file1_exo_name, 'vals_nod_var1'); 
%y-displacement
displ1_y = ncread(file1_exo_name, 'vals_nod_var2'); 
%z-displacement
displ1_z = ncread(file1_exo_name, 'vals_nod_var3'); 
%get last snapshot
displ1_x = displ1_x(:,end); 
displ1_y = displ1_y(:,end); 
displ1_z = displ1_z(:,end); 
%concatenate into a single displacement vector
displ1 = zeros(3*length(displ1_x),1); 
displ1(1:3:end) = displ1_x; 
displ1(2:3:end) = displ1_y; 
displ1(3:3:end) = displ1_z; 

displ{1} = displ0; 
displ{2} = displ1; 

if (schwarz_no == 0)
  %TO ASK ALEJANDRO: Is this right -- to set disp_old to all zeros in the first Schwarz step always? 
  displ_old{1} = zeros(length(displ{1}),1); 
  displ_old{2} = zeros(length(displ{2}),1); 
else
  %read displ_old from file
  displ_old{1} = dlmread('displ0_old'); 
  displ_old{2} = dlmread('displ1_old');
end
 
%The following is based on Alejandro's file FullSchwarz.m 
%specific case of 2 domains
for i=1:2
  displacement_norms(i) = norm(displ{i}); 
  diff = displ{i} - displ_old{i}; 
  difference_norms(i) = norm(diff); 
end

norm_displ = norm(displacement_norms, norm_type); 
norm_difference = norm(difference_norms, norm_type); 

%compute error which will be used to determine if Schwarz has converged.
disp(['      norm_displ = ', num2str(norm_displ)]); 
if (norm_displ > 0.0)
  error = norm_difference / norm_displ; 
else
  error = norm_difference;
end

disp(['      error = ', num2str(error)]);

%debug output
disp('       displ0 = ');
displ{1}
disp('       displ1 = ');
displ{2}

%write new displacements to displ*_old files.
dlmwrite('displ0_old', displ{1}, 'precision', 10); 
dlmwrite('displ1_old', displ{2}, 'precision', 10);
%write new displacements to files with unique names for debugging/diagnosing 
dlmwrite(displ0_current_name, displ{1}, 'precision', 10); 
dlmwrite(displ1_current_name, displ{2}, 'precision', 10); 
%write error to file
dlmwrite('error', error, 'precision', 10);
%write error to file with unique name for debugging/diagnosing 
dlmwrite(error_name, error, 'precision', 10); 





