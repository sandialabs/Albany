fdir_in = '../ncGridSamples/';
fname_geo = 'greenland.nc';  

fdir_out = '../albanyMesh/';
fname_out = 'greenland_2d';
% 
fname_dhdt = fname_geo;
fname_smb = fname_geo; 
fname_flowfact = fname_geo;
ns=1;

%not sure how CISM shift coordinates in general
obs_coords_shift = [-48000, -48000];%[-6000,-6000];

%used to understand where ice is floating
rho_ice = 910; %917;
rho_w = 1028; %1025;
g = 9.81;
pmp_const = 9.7456e-8;
T0 = 273.15;


ice_limit = 10; %[m]

kilo = 1000;

beta_glimmer_scaling = 1;%1/(rho_ice*g*2000/500);

min_beta = 1e-6; %used for the regularized beta



%import mesh. If a field is not present in the mesh, the filed vector will be empty 
s_names = struct('acab', 'acab', 'x1', 'x1', 'y1', 'y1', 'x0', 'x0', 'y0', 'y0','tempr','tempstag', 'flow','flwastag', 'sigma_coord', 'stagwbndlevel','beta', 'beta', 'topg','topg','topgerr','topgerr','thk', 'thk','vx', 'vx', 'vy', 'vy', 'ex','ex', 'ey', 'ey','velnorm','velnorm', 'artm', 'artm', 'dhdt', 'dhdt');
s_geo = netcdf_read( [fdir_in, fname_geo], s_names);

acab= s_geo.acab; x1=s_geo.x1; y1=s_geo.y1;
ex = s_geo.ex; ey=s_geo.ey; vx = s_geo.vx; vy=s_geo.vy;
beta = s_geo.beta(:,:,1); thk = s_geo.thk; topg = s_geo.topg;topgerr = s_geo.topgerr; x0 = s_geo.x0; y0 = s_geo.y0;
beta= beta * beta_glimmer_scaling;
tempr = s_geo.tempr;
velnorm = s_geo.velnorm;
artm = s_geo.artm;
flow = s_geo.flow;
dhdt = s_geo.dhdt;
dhdt_mask = dhdt > -1e6;
dhdt(~dhdt_mask) = 0;

%albany's levels are reversed
sigma_coord = 1-s_geo.sigma_coord(end:-1:1);

velnormStruct = zeros(size(velnorm(:,:,1))+1);
velnormStruct(2:end,2:end) = velnorm(:,:,1);
mask = thk(1:ns:end,1:ns:end)>ice_limit;

clear s_geo;

%% Make sure all coords and fields are consistent
x1 = (x1-obs_coords_shift(1))/kilo;  %convert to km
y1 = (y1-obs_coords_shift(2))/kilo;  %convert to km
x0 = (x0-obs_coords_shift(1))/kilo; %convert to km
y0 = (y0-obs_coords_shift(2))/kilo; %convert to km

x1= x1(1:ns:end,1:ns:end);
y1= y1(1:ns:end,1:ns:end);
x0= x0(1:ns:end,1:ns:end);
y0= y0(1:ns:end,1:ns:end);

% if(~isempty(vx) && ~isempty(vy))
%   vx(vx==vx(1,1))=0;
%   vy(vy==vy(1,1))=0;
%   vx = max(min(vx, max_vel), -max_vel);
%   vy = max(min(vy, max_vel), -max_vel);
% end
% 
% if(~isempty(ex) && ~isempty(ey))
%   ex(ex == ex(1,1))=1e9;
%   ey(ey == ey(1,1))=1e9;
%   ex = max(ex, min_vel_rms);
%   ey = max(ey, min_vel_rms);  
% end

vx = vx(1:ns:end,1:ns:end);
vy = vy(1:ns:end,1:ns:end);
ex = ex(1:ns:end,1:ns:end);
ey = ey(1:ns:end,1:ns:end);


[xi,yi] = meshgrid(x1,y1);

%if(~isempty(sigma_coord))
nTemplevels = size(tempr,3);
nLayers=nTemplevels-2; %nFlowlevels;
nFlowlevels = size(flow,3);

betaStruct = zeros(size(xi'));
if(~isempty(beta))  %shift beta to mesh nodes
  betaStruct(1:length(x0),1:length(y0)) =beta(1:ns:end,1:ns:end)/kilo;
end


velStruct=[];
sigmaVelStruct=[];
thkStruct = thk(1:ns:end,1:ns:end)/kilo;     %convert to km
topgStruct = topg(1:ns:end,1:ns:end)/kilo;   %convert to km
artmStruct = artm(1:ns:end,1:ns:end)+T0;   %convert to Kelvin
topgerrStruct = topgerr(1:ns:end,1:ns:end)/kilo;   %convert to km

%return

if(~isempty(vx) && ~isempty(vy))
  velStruct = cat(3,vx,vy);
end
if(~isempty(ex) && ~isempty(ey))
  sigmaVelStruct = cat(3,ex,ey);
end

if(isempty(topgStruct))
  topgStruct = 0*thkStruct;
end

if(isempty(artmStruct))
  artmStruct = 0*thkStruct;
end

if(isempty(topgerrStruct))
  topgerrStruct = 0*thkStruct;
end


acabStruct = acab(1:ns:end,1:ns:end);  %convert to km/yr
 acabStruct = [];
 acabStruct_RMS = [];
% acab_RMS_mask_not = [];
% if(~isempty(acab))
%   acabStruct =  interp2(Xsmb, Ysmb, smb', xi, yi)';
%   if(~isempty(dhdt))
%     acabStruct = acabStruct - interp2(X, Y, dhdt', xi, yi)';
%     acab_RMS_mask_not =  interp2(X, Y, double(~dhdt_mask)', xi, yi)';
%     acab_RMS_mask_not(isnan(acab_RMS_mask_not)) = 1;
%   end
% end

clear  topg vx vy ex ey acab

%% compute vertices mask and weitght for field reconstruction

mask0 = mask(2:end,2:end);
numElements = sum(sum(mask0));

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


%% import effective temperature (adjusted based on pressure melting point) on elem grid
if(~isempty(tempr))
temprUnstruct=zeros(numElements, nTemplevels); 
  for i=1:nTemplevels    
    il = nTemplevels-i+1;
    % adjusted based on pressure melting point
    tempr2d = min(0,tempr(1:ns:end,1:ns:end,il) + pmp_const*rho_ice*g*(1-sigma_coord(i))*thkStruct*kilo);
    temprUnstruct(:,i) = tempr2d(mask);
  end 
end

%% import flow factor on elem grid
if(~isempty(flow))
    flowUnstruct=zeros(numElements, nFlowlevels); 
    for i=1:nFlowlevels   
        il = nFlowlevels-i+1;
        flow2d = flow(1:ns:end,1:ns:end,il);
        flowUnstruct(:,i) = flow2d(mask);
    end
end


%% reconstruct velocity on nodal grid
if(~isempty(velStruct))
    for i=1:2
        vel2d = velStruct(:,:,i);
        vel2d(~mask) = 0;
        vel2d(1:end-1,1:end-1) = vel2d(1:end-1,1:end-1)+vel2d(2:end,1:end-1)+vel2d(1:end-1,2:end)+vel2d(2:end,2:end);
        vel2d(I) = vel2d(I)./vertex_weight(I);
    end
end

%% reconstruct velocity RMS on nodal grid
if(~isempty(sigmaVelStruct))
    for i=1:2
        sigmaVel2d = sigmaVelStruct(:,:,i);
        sigmaVel2d(~mask) = 0;
        sigmaVel2d(1:end-1,1:end-1) = sigmaVel2d(1:end-1,1:end-1)+sigmaVel2d(2:end,1:end-1)+sigmaVel2d(1:end-1,2:end)+sigmaVel2d(2:end,2:end);
        sigmaVel2d(I) = sigmaVel2d(I)./vertex_weight(I);
    end
    
    % ad hoc interpolation
    acabStruct_RMS = [];
    %acabStruct_RMS = 0.50+0.495*2/pi*atan(5*log(max(0.001,velnormStruct(:,:,1))/100));
    %acabStruct_RMS(logical(acab_RMS_mask_not)) = 1e6;
end



%% reconstruct acab on nodal grid
if(~isempty(acabStruct))
    acabStruct(~mask) = 0;
    acabStruct(1:end-1,1:end-1) = acabStruct(1:end-1,1:end-1)+acabStruct(2:end,1:end-1)+acabStruct(1:end-1,2:end)+acabStruct(2:end,2:end);
    acabStruct(I) = acabStruct(I)./vertex_weight(I);
end

%% reconstruct acab on nodal grid
if(~isempty(acabStruct_RMS))
    acabStruct_RMS(~mask) = 0;
    acabStruct_RMS(1:end-1,1:end-1) = acabStruct_RMS(1:end-1,1:end-1)+acabStruct_RMS(2:end,1:end-1)+acabStruct_RMS(1:end-1,2:end)+acabStruct_RMS(2:end,2:end);
    acabStruct_RMS(I) = acabStruct_RMS(I)./vertex_weight(I);
end

%% reconstruct topg on nodal grid
if(~isempty(topgStruct))
    topgStruct(~mask) = 0;
    topgStruct(1:end-1,1:end-1) = topgStruct(1:end-1,1:end-1)+topgStruct(2:end,1:end-1)+topgStruct(1:end-1,2:end)+topgStruct(2:end,2:end);
    topgStruct(I) = topgStruct(I)./vertex_weight(I);
end

%% reconstruct topg on nodal grid
if(~isempty(topgerrStruct))
    topgerrStruct(~mask) = 0;
    topgerrStruct(1:end-1,1:end-1) = topgerrStruct(1:end-1,1:end-1)+topgerrStruct(2:end,1:end-1)+topgerrStruct(1:end-1,2:end)+topgerrStruct(2:end,2:end);
    topgerrStruct(I) = topgerrStruct(I)./vertex_weight(I);
end

%% reconstruct thickness on nodal grid
thkStruct(~mask) = 0;%2 = zeros(size(thkStruct));
thkStruct(1:end-1,1:end-1) = thkStruct(1:end-1,1:end-1)+thkStruct(2:end,1:end-1)+thkStruct(1:end-1,2:end)+thkStruct(2:end,2:end);
thkStruct(I) = thkStruct(I)./vertex_weight(I);

 
disp('mesh data imported');


[coords, quad, new2old_vertex, old2new_vertex, bd_points, boundary_edges] = nc_to_msh_cut_quad(x1, y1, vertex_mask, mask0, [fdir_out,fname_out], false);


disp('computed msh mesh from nc');


thickness = interp2(xi, yi, thkStruct', coords(:,1), coords(:,2)); %convert dofs from the structured to the unstructured mesh
bedrock = interp2(xi, yi, topgStruct', coords(:,1), coords(:,2)); %convert to the unstructured mesh
thicknessRMS = interp2(xi, yi, topgerrStruct', coords(:,1), coords(:,2)); %convert to the unstructured mesh
beta = interp2(xi, yi, betaStruct', coords(:,1), coords(:,2));  %convert to the unstructured mesh
elevation =  max(thickness+bedrock, thickness*(1-rho_ice/rho_w)); % make sure ice satisfies floating condition when in the water

acab=[]; acab_RMS=[];
if(~isempty(acabStruct))
    if(~isempty(acabStruct_RMS))
        acab = interp2(xi, yi, acabStruct', coords(:,1), coords(:,2));  %convert to the unstructured mesh
        acab_RMS = interp2(xi, yi, acabStruct_RMS', coords(:,1), coords(:,2));  %convert to the unstructured mesh
    end
end


%% compute boundary edges, flag=4 for floating, 3 for grounded
boundary_edges(:,3) = 3+((thickness(boundary_edges(:,1)) < -rho_w/rho_ice*bedrock(boundary_edges(:,1))) | (thickness(boundary_edges(:,2)) < -rho_w/rho_ice*bedrock(boundary_edges(:,2))));

%% plot boundary edges
%for i=1:size(boundary_edges,1)
%    if(boundary_edges(i,3)-3) %red edges are floating, blue are grounded
%      plot([coords(boundary_edges(i,1),1), coords(boundary_edges(i,2),1)], [coords(boundary_edges(i,1),2), coords(boundary_edges(i,2),2)],'r-', 'linewidth' ,2);
%    else
%      plot([coords(boundary_edges(i,1),1), coords(boundary_edges(i,2),1)], [coords(boundary_edges(i,1),2), coords(boundary_edges(i,2),2)],'b-');
%    end
%hold on
%end

boundary_edges = boundary_edges(boundary_edges(:,3) == 4, :);


%% print mesh
print_msh_quad([fdir_out,fname_out,'.quad'], coords, quad, boundary_edges); 


%% interpolate fields on vertices belonging to the mask 
vel = zeros(size(coords,1), size(velStruct,3));
sigmaVel = zeros(size(coords,1), size(velStruct,3));


if(~isempty(velStruct) && ~isempty(sigmaVelStruct))
  for i=1:size(velStruct,3)
    vel(:,i) = interp2(xi, yi, velStruct(:,:,i)', coords(:,1), coords(:,2)); %convert to the unstructured mesh
    sigmaVel(:,i) = interp2(xi, yi, sigmaVelStruct(:,:,i)', coords(:,1), coords(:,2));
  end
end

vel = reshape(vel, numel(vel), 1);

sigmaVel = reshape(sigmaVel, numel(sigmaVel),1);

if(~isempty(tempr))
    tempr = reshape(temprUnstruct+T0, numel(temprUnstruct), 1); %convert to kelvin units
end


if(~isempty(flow))
  flow = reshape(flowUnstruct*kilo^4, numel(flowUnstruct), 1); %rescale using Albany units
end

%threshold sigmas (make sure they are not too small)
sigmaVel = max(max(sigmaVel, 0.05*abs(vel)),0.05);
thicknessRMS = max(0.1,thicknessRMS');


%% write fieds into ascii files
numComponents = size(thickness,1);

if(~isempty(tempr))
  fid = fopen ( [fdir_out, 'temperature.ascii'], 'wt' );
  fprintf(fid,'%d %d\n', numElements, nTemplevels);
  fprintf(fid,'%g\n', sigma_coord);
  fprintf(fid,'%g\n', tempr');
  fclose(fid);
end

if(~isempty(flow))
  fid = fopen ( [fdir_out, 'flow_rate.ascii'], 'wt' );
  fprintf(fid,'%d %d\n', numElements, nFlowlevels);
  fprintf(fid,'%g\n', sigma_coord(2:end-1));
  fprintf(fid,'%g\n', flow');
  fclose(fid);
end

fid = fopen ( [fdir_out, 'surface_velocity.ascii'], 'wt' );
fprintf(fid,'%d %d\n', numComponents, 2);
fprintf(fid,'%8g\n', vel');
fclose(fid);

fid = fopen ( [fdir_out, 'velocity_RMS.ascii'], 'wt' );
fprintf(fid,'%d %d\n', numComponents, 2);
fprintf(fid,'%g\n', sigmaVel');
fclose(fid);

fid = fopen ( [fdir_out, 'surface_height.ascii'], 'wt' );
fprintf(fid,'%d\n', numComponents);
fprintf(fid,'%g\n', elevation');
fclose(fid);

fid = fopen ( [fdir_out, 'thickness.ascii'], 'wt' );
fprintf(fid,'%d\n', numComponents);
fprintf(fid,'%g\n', thickness');
fclose(fid);


fid = fopen ( [fdir_out, 'basal_friction.ascii'], 'wt' );
fprintf(fid,'%d\n', numComponents);
fprintf(fid,'%g\n', beta');
fclose(fid);

fid = fopen ( [fdir_out, 'basal_friction_reg.ascii'], 'wt' );
fprintf(fid,'%d\n', numComponents);
fprintf(fid,'%g\n', max(beta,min_beta)');
fclose(fid);

% %used for initialization
% fid = fopen ( [fdir_out, 'basal_friction_log.ascii'], 'wt' );
% fprintf(fid,'%d\n', length(beta));
% fprintf(fid,'%g\n', log(max(7+0.*beta,1e-16))');%log(7+0.*beta'));
% fclose(fid);


fid = fopen ( [fdir_out, 'surface_mass_balance.ascii'], 'wt' );
fprintf(fid,'%d\n', length(acab)); %numComponents);
fprintf(fid,'%g\n', acab');
fclose(fid);

fid = fopen ( [fdir_out, 'surface_mass_balance_RMS.ascii'], 'wt' );
fprintf(fid,'%d\n', length(acab_RMS)); %numComponents);
fprintf(fid,'%g\n', acab_RMS');
fclose(fid);

fid = fopen ( [fdir_out, 'thickness_RMS.ascii'], 'wt' );
fprintf(fid,'%d\n', length(thicknessRMS)); %numComponents);
fprintf(fid,'%g\n', thicknessRMS');
fclose(fid);

return




