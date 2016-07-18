function [coords, Quad, new2old_vertex, old2new_vertex, bd_points,  boundary_edges, boundary_label] = nc_to_msh_cut_quad(x1, y1, mask2d_bool, mask2d_elem,fname, print)

keep_vertex_NW = false(size(mask2d_bool)-1);
keep_vertex_NE = false(size(mask2d_bool)-1);
keep_vertex_SW = false(size(mask2d_bool)-1);
keep_vertex_SE = false(size(mask2d_bool)-1);

thk_bool_NW = mask2d_bool(1:end-1,1:end-1);
thk_bool_NE = mask2d_bool(1:end-1,2:end);
thk_bool_SW = mask2d_bool(2:end,1:end-1);
thk_bool_SE = mask2d_bool(2:end,2:end);

%pcolor(x1(1:end-1), y1(1:end-1)', double(thk_bool_NW + thk_bool_NE + thk_bool_SE + thk_bool_SW >= 4)'); shading interp
if(~isempty(mask2d_elem))
    I_keep_quad = find(mask2d_elem == 1);
else
    I_keep_quad = find(thk_bool_NW & thk_bool_NE & thk_bool_SE & thk_bool_SW);
end


%I_keep_quad = find(thk_bool_NW + thk_bool_NE + thk_bool_SE + thk_bool_SW >= 4); % ~= 0

keep_vertex_NW(I_keep_quad) = true;
keep_vertex_NE(I_keep_quad) = true;
keep_vertex_SW(I_keep_quad) = true;
keep_vertex_SE(I_keep_quad) = true;

keep_vertex = false(size(mask2d_bool));
keep_vertex(1:end-1, 1:end-1) = keep_vertex_NW;
keep_vertex(1:end-1, 2:end) = keep_vertex(1:end-1, 2:end) | keep_vertex_NE;
keep_vertex(2:end, 1:end-1) = keep_vertex(2:end, 1:end-1) | keep_vertex_SW;
keep_vertex(2:end, 2:end) = keep_vertex(2:end, 2:end) | keep_vertex_SE;
clear keep_vertex_NW keep_vertex_NE keep_vertex_SW keep_vertex_SE;

new2old_vertex = find(keep_vertex);
old2new_vertex = zeros(numel(keep_vertex),1);
old2new_vertex(new2old_vertex) = (1:length(new2old_vertex))';
clear keep_vertex

num_row_vert = size(mask2d_bool,1);
num_row_quad = size(mask2d_bool,1)-1;

Quad = zeros(length(I_keep_quad),4);
k = 1;
for iq = I_keep_quad'
    i = mod(iq-1, num_row_quad)+1;
    j = floor((iq-1)/num_row_quad)+1;
    idNW = old2new_vertex(i + (j-1)*num_row_vert);
    idNE = old2new_vertex(i + j*num_row_vert);
    idSW = old2new_vertex(i + 1 + (j-1)*num_row_vert);
    idSE = old2new_vertex(i + 1 + j*num_row_vert);  
    Quad(k,:) = [idSW, idSE, idNE, idNW];
    k = k + 1;
end
Quad = Quad(1:k-1,:);

i = mod(new2old_vertex-1, num_row_vert)+1;
j = floor((new2old_vertex-1)/num_row_vert)+1;
coords = [x1(i),y1(j)];

%temp_rsh = reshape(temp, [size(temp,1)*size(temp,2),size(temp,3)]);
%temperature = reshape(temp_rsh(new2old_vertex,:), [length(new2old_vertex)*size(temp,3),1]);

edges = zeros(4*size(Quad,1),2);

points = zeros(4*size(Quad,1),1);
for i=0:3
  points(i+1:4:end,:) = Quad(:,i+1);
end

size(unique(points))


for i=0:3
  edges(i+1:4:end,:) = Quad(:,[mod(i,4)+1, mod(i+1,4)+1]);
end
sortedEdges = sort(edges,2);
[temp, Iu] = unique(sortedEdges,'rows');  size(temp)
I_repeated = true(size(edges,1),1); I_repeated(Iu) = false; clear Iu;
[~, I_bdEdges] = setdiff(sortedEdges, sortedEdges(I_repeated,:),'rows'); 
clear temp sortedEdges;
boundary_edges = edges(I_bdEdges, :);

bd_points = unique([boundary_edges(:,1); boundary_edges(:,2)]);

vertex_label = zeros(1,size(coords,1));
vertex_label(bd_points) = 1;
coords = [coords, vertex_label'];
quad_label = ones(1, size(Quad,1),1);
boundary_label = ones(size(boundary_edges,1),1);
% mesh_write( [fname,'_2d.mesh'], 2, size(coords,1), coords', ...
%     vertex_label, size(boundary_edges,1), boundary_edges', boundary_label', size(Quad,1), Quad', ...
%     quad_label, 0, [], ...
%     [], 0, [], [], 0, [], [] );

%output = fopen ( [fname,'_2d.bb'], 'wt' );
%fprintf (output, '%d %d %d %d\n', [2; 1; length(thickness); 2]);
%fprintf (output, '%f\n', thickness);
%fclose(output);

if(print)
  fid = fopen ( [fname,'.msh'], 'wt' );
  fprintf(fid,'Quadrilateral 4\n');
  fprintf(fid,'%d  %d %d\n', size(coords,1), size(Quad,1), size(boundary_edges,1));
  fprintf(fid,'%g  %g %d\n', coords');  
  fprintf(fid,'%d  %d %d %d\n',[Quad, quad_label']');
  fprintf(fid,'%d  %d %d\n',[boundary_edges, boundary_label]');
  fclose(fid);
end

                
end
            
