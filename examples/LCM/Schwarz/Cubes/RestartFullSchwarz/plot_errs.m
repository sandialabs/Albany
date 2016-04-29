
close all; clear all; 


num_load_steps = 11; 

for i=1:num_load_steps
    error_filenames = strcat('error_load',num2str(i-1),'_filenames'); 
    errors = strcat('error_load',num2str(i-1),'_values'); 
    err_order=dlmread(error_filenames)+1; 
    [X,I] = sort(err_order); 
    err = dlmread(errors); 
    figure(); 
    semilogy(err_order(I), err(I), '-o'); 
    xlabel('schwatz iter #'); 
    ylabel('schwarz error'); 
    title(['load step = ', num2str(i), ': ', num2str(length(err)), ' schwarz iters']); 
    num_schwarz_iter(i) = length(err); 
    errs{i} = err(I); 
end

num_schwarz_iter