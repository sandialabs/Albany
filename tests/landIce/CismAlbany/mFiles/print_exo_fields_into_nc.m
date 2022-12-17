fdir_inout = '../';
exo_fname_in = 'greenland_standalone-albanyT.exo';
exo_fname2_in = 'greenland_cism-albanyT.exo';

nc_fname_inout = 'greenland_standalone-albanyT.nc'

kilo = 1000;

rho_i = 910;
rho_w = 1028;

ice_limit = 10; %[m]

nLevels=11;  %careful! this needs to be compatible with the grids

isAlbanyMeshOrderColumnWise = false;

rise=0;

%% import fields of exo mesh

s_exo_names = struct('basal_friction','basal_friction', 'Velx', 'solution_x', 'Vely', 'solution_y', 'sh', 'surface_height', 'thk', 'thickness');

s_geo = exo_read( [fdir_inout, exo_fname_in], s_exo_names);

%% extract fields from s_geo struct

exo_beta = s_geo.basal_friction(:,end);
exo_Velx = s_geo.Velx(:,end);
exo_Vely = s_geo.Vely(:,end);
exo_sh = s_geo.sh(:,end);
exo_thk = s_geo.thk(:,end);

size3d = length(exo_beta);
size2d = size3d/nLevels;


%if ordering is clumnwise
exo_beta = reshape(exo_beta, nLevels, size2d)'; 
exo_thk = reshape(exo_thk, nLevels, size2d)';
exo_sh = reshape(exo_sh, nLevels, size2d)'+rise;
exo_Velx = reshape(exo_Velx, nLevels, size2d)';
exo_Vely = reshape(exo_Vely, nLevels, size2d)';

% fid = fopen ( [fdir_inout, 'thickness.ascii'], 'wt' );
% fprintf(fid,'%d\n', size2d);
% fprintf(fid,'%g\n', exo_thk(:,1)');
% fclose(fid);
% 
% fid = fopen ( [fdir_inout, 'surface_elevation.ascii'], 'wt' );
% fprintf(fid,'%d\n', size2d);
% fprintf(fid,'%g\n', exo_thk(:,1)');
% fclose(fid);
% 
% fid = fopen ( [fdir_inout, 'basal_friction_log.ascii'], 'wt' );
% fprintf(fid,'%d\n', size2d);
% fprintf(fid,'%g\n', exo_beta(:,1)');
% fclose(fid);
% 
% fid = fopen ( [fdir_inout, 'basal_friction.ascii'], 'wt' );
% fprintf(fid,'%d\n', size2d);
% fprintf(fid,'%g\n', exp(exo_beta(:,1))');
% fclose(fid);



%% compute vertices mask and weitght for field reconstruction

%import netcdf fields
s_names = struct('x1','x1', 'y1', 'y1','thk', 'thk', 'topg', 'topg', 'beta', 'beta');
s_geo_nc = netcdf_read( [fdir_inout,nc_fname_inout], s_names);
x1 = s_geo_nc.x1; y1 = s_geo_nc.y1; 


%compute mask and vertices mask
mask = s_geo_nc.thk > ice_limit;
mask0 = mask(2:end,2:end);

vertex_mask = false(size(mask));
vertex_mask(1:end-1,1:end-1) = mask0; 
vertex_mask(1:end-1,2:end) = vertex_mask(1:end-1,2:end) | mask0;
vertex_mask(2:end,1:end-1) = vertex_mask(2:end,1:end-1) | mask0;
vertex_mask(2:end,2:end) = vertex_mask(2:end,2:end) | mask0;

vertex_weight = zeros(size(mask)); 
vertex_weight(1:end-1,1:end-1) = mask0; 
vertex_weight(1:end-1,2:end) = vertex_weight(1:end-1,2:end) + mask0;
vertex_weight(2:end,1:end-1) = vertex_weight(2:end,1:end-1) + mask0;
vertex_weight(2:end,2:end) = vertex_weight(2:end,2:end) + mask0;

I = (vertex_weight>0);


[coords, quad, new2old_vertex, old2new_vertex, bd_points, boundary_edges] = nc_to_msh_cut_quad(x1, y1, vertex_mask, mask0, [fdir_inout,nc_fname_inout], false);



disp('computed msh mesh from nc');

%interpolate thickness and bedrock topography
thkStruct = zeros(size(vertex_mask));
thkStruct2d = zeros(size(vertex_mask));
thkStruct2d(new2old_vertex) = exo_thk(:,1)*kilo;
thkStruct2d_orig = thkStruct2d;
thkStruct2d = 0.25*(thkStruct2d(1:end-1,1:end-1)+thkStruct2d(1:end-1,2:end)+thkStruct2d(2:end,1:end-1)+thkStruct2d(2:end,2:end));
thkStruct(mask) = thkStruct2d(mask0);


topgStruct = s_geo_nc.topg + s_geo_nc.thk - thkStruct;
% topgStruct2d = s_geo_nc.topg+rise*kilo;
% topgStruct = topgStruct2d;
% topgStruct2d(new2old_vertex) = (exo_sh(:,1)- exo_thk(:,1))*kilo;
% topgStruct2d = 0.25*(topgStruct2d(1:end-1,1:end-1)+topgStruct2d(1:end-1,2:end)+topgStruct2d(2:end,1:end-1)+topgStruct2d(2:end,2:end));
% grounded = topgStruct2d(mask0) > -rho_i/rho_w*thkStruct2d(mask0);
% topgStruct(mask) = grounded .* (topgStruct2d(mask0) + rise*kilo) + (~grounded) .* topgStruct(mask);

thkStructAlb = zeros(size(thkStruct));
thkStructAlb(1:end-1,1:end-1) = thkStruct(1:end-1,1:end-1)+thkStruct(2:end,1:end-1)+thkStruct(1:end-1,2:end)+thkStruct(2:end,2:end);
thkStructAlb(I) = thkStructAlb(I)./vertex_weight(I);
thkStructAlb(~I) = 0;


%export velocity and basal friciton
velStruct = zeros(size(vertex_mask));
velxStruct = zeros(size(vertex_mask,1)-1, size(vertex_mask,2)-1, nLevels);
velyStruct = zeros(size(vertex_mask,1)-1, size(vertex_mask,2)-1, nLevels);

betaStruct2d = zeros(size(vertex_mask));    
betaStruct2d(1:end-1,1:end-1) = s_geo_nc.beta;
betaStruct2d(new2old_vertex) = exo_beta(:,1)*kilo;
betaStruct = betaStruct2d(1:end-1,1:end-1);
%betaStruct = extend_field(betaStruct2d(1:end-1,1:end-1),50);
    
for i=1:nLevels
    velStruct(new2old_vertex) = exo_Velx(:,nLevels+1-i);
    velxStruct(:,:,i) =  velStruct(1:end-1,1:end-1); %0.25*( myVelStruct(1:end-2,1:end-2) +   myVelStruct(2:end-1,1:end-2) + myVelStruct(1:end-2,2:end-1) +  myVelStruct(2:end-1,2:end-1));%myVelStruct(1:end-1,1:end-1);
    velStruct(new2old_vertex) = exo_Vely(:,nLevels+1-i);
    velyStruct(:,:,i) = velStruct(1:end-1,1:end-1);
end

  ncwrite([fdir_inout,nc_fname_inout], 'beta', betaStruct);
  ncwrite([fdir_inout,nc_fname_inout], 'uvel', velxStruct);
  ncwrite([fdir_inout,nc_fname_inout], 'vvel', velyStruct);
  ncwrite([fdir_inout,nc_fname_inout], 'velnorm', sqrt(velxStruct.^2+velyStruct.^2));
  ncwrite([fdir_inout,nc_fname_inout], 'thk', thkStruct);
  ncwrite([fdir_inout,nc_fname_inout], 'topg',topgStruct);


% warning
disp(['Maximum difference in thickess between the standalone Albany one and the one seen by Albany when called by CISM is: ', num2str(max(max(abs(thkStructAlb'-thkStruct2d_orig'))))])
  
  
  
% [X,Y] =meshgrid(x1,y1);
% figure(1)
% pcolor(X,Y,thkStruct2d_orig'); colorbar; shading interp; title('Albany H_{opt}');
% figure(2)
% pcolor(X,Y, thkStruct'); colorbar; shading interp; title('H_{opt} interpolated to nc');
% %figure(3)
% %pcolor(X,Y, thkStruct'-s_geo_nc.thk'); colorbar; shading interp; 
% figure(3)
% pcolor(X,Y, thkStructAlb'); colorbar; shading interp; title('H_{opt} interpolated back to Albany');
% figure(4)
% pcolor(X,Y, thkStructAlb'-thkStruct2d_orig'); colorbar; shading interp; title('difference between oroginal H_{opt} and the one coming from nc');
