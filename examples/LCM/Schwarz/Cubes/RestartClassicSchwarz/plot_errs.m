
close all; clear all; 


num_load_steps = 11; 

for i=1:num_load_steps
    error_filenames = strcat('error_load',num2str(i-1),'_filenames'); 
    errors = strcat('error_load',num2str(i-1),'_values'); 
    err_order=dlmread(error_filenames)+1; 
    err = dlmread(errors); 
    figure(); 
    plot(err_order, err, 'o'); 
    xlabel('schwatz iter #'); 
    ylabel('schwarz error'); 
    title(['load step', num2str(i)]); 
    num_schwarz_iter(i) = length(err); 
end

num_schwarz_iter