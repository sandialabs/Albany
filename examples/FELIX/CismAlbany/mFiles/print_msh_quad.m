function print_msh_quad(fname, coords, quad, edg)

fid = fopen ( [fname,'.msh'], 'wt' );
fprintf(fid,'Quadrilateral 4\n');
fprintf(fid,'%d  %d %d\n', size(coords,1), size(quad,1), size(edg,1));
fprintf(fid,'%f  %f %d\n', coords');
fprintf(fid,'%d  %d %d %d %d\n',[quad, ones(size(quad,1),1)]');
fprintf(fid,'%d  %d %d\n', edg');
fclose(fid);
