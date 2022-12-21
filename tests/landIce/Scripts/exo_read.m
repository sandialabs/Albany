function  [fields_struct] = exo_read( filename, names_struct )

%*****************************************************************************80
%
%% NECDF_READ reads the NETCDF file containing the GIS data.
%
%  Discussion:
%
%    For what I suppose are perfectly good reasons, MATLAB returns
%    multiply-dimensioned arrays with the dimensioning in the reverse
%    order of what the internal NETCDF documentation would suggest.
%    Thus, an array which is supposed to be dimensioned THK(TIME,Y1,X1)
%    will be returned as THK(X1,Y1,TIME).
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    6 December 2014
%
%  Authors:
%
%    Mauro Perego, John Burkardt
%
%  Reference:
%
%    Russ Rew, Glenn Davis, Steve Emmerson, Harvey Davies, Ed Hartne,
%    The NetCDF User's Guide,
%    Unidata Program Center, March 2009.
%
%  Parameters:
%
%    Input, string FILENAME, the name of the file to be read.
%    Ordinarily, the name should include the extension ".nc".
%   
%    Input, struct names_struct. The values of the struct fields are the
%    names of the 2d field to be pulled from the netcdf file filename.
%
%    Output, struct containing the same fields of the input struct, but
%    having values equal to the 2d field pulled from the netcdf file.
%

%
%  Open the file.
%

  s_exo_names = struct('node_variables', 'name_nod_var',  'elem_variables', 'name_elem_var', 'global_variables', 'name_glo_var');
  names = netcdf_read( filename, s_exo_names);
  node_variables = cellstr(names.node_variables');
  elem_variables = cellstr(names.elem_variables');
  global_variables = cellstr(names.global_variables');
  
 
  s_names = fieldnames(names_struct);
  for i=1:length(s_names)
    index = find(strcmp(node_variables, names_struct.(s_names{i})));
    if(~isempty(index))
        names_struct.(s_names{i}) = ['vals_nod_var',int2str(index)];
    else
        index = find(strcmp(elem_variables, names_struct.(s_names{i})));
        if(~isempty(index))
            names_struct.(s_names{i}) = ['vals_elem_var',int2str(index),'eb1'];
        else
            index = find(strcmp(global_variables, names_struct.(s_names{i})));
            if(~isempty(index))
                names_struct.(s_names{i}) = ['vals_glo_var'];
            end
        end
    end
  end
  
  fields_struct = netcdf_read( filename, names_struct );
  return
end