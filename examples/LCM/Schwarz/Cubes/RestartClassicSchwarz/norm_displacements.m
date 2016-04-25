
%Note that this function assumes 2 domain.  Can be generalized
%to an arbitrary number of domains. 

%Another assumption here is that disp_x, disp_y and disp_z are
%vals_nod_var1, vals_nod_var2, and vals_nod_var3 in the *exo file,
%respectively.  If they are not, code needs to be modified.

function[error] = norm_displacements(file0_exo_name, file1_exo_name) 

%file0_exo_name = 'cube0_restart_1.exo';
%file1_exo_name = 'cube1_restart_1.exo'; 

%Here we hard-code 2-norm.  norm_type could be made an input argument.
norm_type = 2; 

%cube0
%x-displacement
disp0_x = ncread(file0_exo_name, 'vals_nod_var1'); 
%y-displacement
disp0_y = ncread(file0_exo_name, 'vals_nod_var2'); 
%z-displacement
disp0_z = ncread(file0_exo_name, 'vals_nod_var3'); 
%get last snapshot
disp0_x = disp0_x(:,end); 
disp0_y = disp0_y(:,end); 
disp0_z = disp0_z(:,end); 
%concatenate into a single displacement vector
disp0 = zeros(3*length(disp0_x),1); 
disp0(1:3:end) = disp0_x; 
disp0(2:3:end) = disp0_y; 
disp0(3:3:end) = disp0_z; 

%cube1
%x-displacement
disp1_x = ncread(file1_exo_name, 'vals_nod_var1'); 
%y-displacement
disp1_y = ncread(file1_exo_name, 'vals_nod_var2'); 
%z-displacement
disp1_z = ncread(file1_exo_name, 'vals_nod_var3'); 
%get last snapshot
disp1_x = disp1_x(:,end); 
disp1_y = disp1_y(:,end); 
disp1_z = disp1_z(:,end); 
%concatenate into a single displacement vector
disp1 = zeros(3*length(disp1_x),1); 
disp1(1:3:end) = disp1_x; 
disp1(2:3:end) = disp1_y; 
disp1(3:3:end) = disp1_z; 

disp{1} = disp0; 
disp{2} = disp1; 

disp0_old = dlmread('disp0_old'); 
disp1_old = dlmread('disp1_old'); 
disp_old{1} = disp0_old; 
disp_old{2} = disp1_old; 

%The following is based on Alejandro's file FullSchwarz.m 
%specific case of 2 domains
for i=1:2
    displacement_norms(i) = norm(disp{i}); 
    diff = disp{i} - disp_old{i}; 
    difference_norms(i) = norm(diff); 
end

norm_disp = norm(displacement_norms, norm_type); 
norm_difference = norm(difference_norms, norm_type); 

%compute error which will be used to determine if Schwarz has converged.
%Currently, error is returned by this function.
norm_disp
if (norm_disp > 0.0)
    error = norm_difference / norm_disp; 
    norm_difference
    norm_difference / norm_disp
    error
else
    error = norm_difference;
end

%write new displacements to disp*_old files.
dlmwrite('disp0_old', disp{1}); 
dlmwrite('disp1_old', disp{2}); 






