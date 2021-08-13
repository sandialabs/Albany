function [mat,map] = readTrilinosMatrix(matFile, mapFile)

% Read matrixmarket files produced by Trilinos (with non-contiguous ids) and
% convert them into a map and a contiguous id matrix. 
%
% Notes:
%   1) This is pretty fragile. Assumes that the first 8 lines of the mapFile are
%      comments followed by the 9th line which contains size information. Assumes
%      that the first line of the matFile is a comment followed by the 2nd line
%      containing size information. If the last sprintf command adds extra spaces
%      when printing integers, the substitution might fail. This hasn't happened so far ... 
%
%   2) On exit, this function leaves a file 'tempConvert.mm' which is a copy of the
%      matrix file, but with the larger sizes in the header for mmread(). 
%
%   3) Assume mmread is available. 

cmd = sprintf('!/bin/rm -f tempConvert.mm; cp %s tempConvert.mm',mapFile);
eval(cmd);
cmd = sprintf('!sed -i -e 1,9d tempConvert.mm');
eval(cmd);

map = load('tempConvert.mm');
map = map(1:2:end); map = map+1;
smallSize = length(map);
largeSize= max(map);

cmd = sprintf('!/bin/rm -f tempConvert.mm; cp %s tempConvert.mm',matFile);
eval(cmd);
cmd = sprintf('!sed -i -e ''1,2s/%d %d /%d %d /'' tempConvert.mm',smallSize,smallSize,largeSize,largeSize);
eval(cmd);
mat=mmread('tempConvert.mm');  mat=mat(map,map);
