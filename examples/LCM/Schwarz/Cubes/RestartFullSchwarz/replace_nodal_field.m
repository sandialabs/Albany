


function[] = replace_nodal_fields(target_file, source_file, target_indices, source_indices); 

numtgts = length(target_indices); 
numsrcs = length(source_indices); 

target_file
source_file

%TODO: add error check if numtgts \neq numsrcs 

%read node vars corresponding to source_indices from source_file
for i=1:numsrcs
  name_src = strcat('vals_nod_var', num2str(source_indices(i))); 
  srcvar{i} = ncread(source_file, name_src);
  name_tgt = strcat('vals_nod_var', num2str(target_indices(i))); 
  ncwrite(target_file, name_tgt, srcvar{i}); 
end



