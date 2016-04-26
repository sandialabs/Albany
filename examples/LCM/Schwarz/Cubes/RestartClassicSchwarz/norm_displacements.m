
%Note that this function assumes 2 domain.  Can be generalized
%to an arbitrary number of domains. 

%Another assumption here is that displ_x, displ_y and displ_z are
%vals_nod_var1, vals_nod_var2, and vals_nod_var3 in the *exo file,
%respectively.  If they are not, code needs to be modified.

%Input: Schwarz step number, step_no (int) 
function[] = norm_displlacements(step_no) 

file0_exo_name = strcat('cube0_restart_',num2str(step_no),'.exo');
file1_exo_name = strcat('cube1_restart_',num2str(step_no),'.exo');

step_no
file0_exo_name
file1_exo_name

%Here we hard-code 2-norm.  norm_type could be made an input argument.
norm_type = 2; 

%cube0
%x-displlacement
displ0_x = ncread(file0_exo_name, 'vals_nod_var1'); 
%y-displlacement
displ0_y = ncread(file0_exo_name, 'vals_nod_var2'); 
%z-displlacement
displ0_z = ncread(file0_exo_name, 'vals_nod_var3'); 
%get last snapshot
displ0_x = displ0_x(:,end); 
displ0_y = displ0_y(:,end); 
displ0_z = displ0_z(:,end); 
%concatenate into a single displlacement vector
displ0 = zeros(3*length(displ0_x),1); 
displ0(1:3:end) = displ0_x; 
displ0(2:3:end) = displ0_y; 
displ0(3:3:end) = displ0_z; 

%cube1
%x-displlacement
displ1_x = ncread(file1_exo_name, 'vals_nod_var1'); 
%y-displlacement
displ1_y = ncread(file1_exo_name, 'vals_nod_var2'); 
%z-displlacement
displ1_z = ncread(file1_exo_name, 'vals_nod_var3'); 
%get last snapshot
displ1_x = displ1_x(:,end); 
displ1_y = displ1_y(:,end); 
displ1_z = displ1_z(:,end); 
%concatenate into a single displlacement vector
displ1 = zeros(3*length(displ1_x),1); 
displ1(1:3:end) = displ1_x; 
displ1(2:3:end) = displ1_y; 
displ1(3:3:end) = displ1_z; 

displ{1} = displ0; 
displ{2} = displ1; 

if (step_no == 0)
  %if it's the first step, set error = 1 so that code continues
  %TODO: check with Alejandro what he does.
  error = 1;  
else
  %The following is based on Alejandro's file FullSchwarz.m 
  %specific case of 2 domains
  %read displ_old from file
  displ_old{1} = dlmread('displ0_old'); 
  displ_old{2} = dlmread('displ1_old'); 
  for i=1:2
    displlacement_norms(i) = norm(displ{i}); 
    diff = displ{i} - displ_old{i}; 
    difference_norms(i) = norm(diff); 
  end

  norm_displ = norm(displlacement_norms, norm_type); 
  norm_difference = norm(difference_norms, norm_type); 

  %compute error which will be used to determine if Schwarz has converged.
  norm_displ
  if (norm_displ > 0.0)
    error = norm_difference / norm_displ; 
  else
    error = norm_difference;
  end
end

disp(['error = ', num2str(error)]); 
%write new displlacements to displ*_old files.
dlmwrite('displ0_old', displ{1}); 
dlmwrite('displ1_old', displ{2}); 
%write error to file
dlmwrite('error', error); 





