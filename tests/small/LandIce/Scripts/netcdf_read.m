function  [fields_struct] = netcdf_read( filename, names_struct )

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
  ncid = netcdf.open ( filename, 'NOWRITE' );
%
%  Read the NETCDF counts.
%
  [ ~, nvars] = netcdf.inq ( ncid );

 
%
%  Read the data by getting, and then specifying the variable index.
%

s_names = fieldnames(names_struct);

for i=1:length(s_names)
    fields_struct.(s_names{i}) = [];
end

for i=0 : nvars - 1
  varName = netcdf.inqVar(ncid,i);
  
  for j=1:length(s_names)
    if strcmp(varName,names_struct.(s_names{j}))
        fields_struct.(s_names{j}) =  ncread(filename,varName);% netcdf.getVar ( ncid, i);
    end
  end
end
    
%
%  Close the file.
%
  netcdf.close ( ncid );

  return
end