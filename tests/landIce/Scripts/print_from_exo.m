
fdir_in = './';
exo_fname_in = 'humboldt_full.exo';

disregard_coord_shifts = true;

%% import fields of exo mesh

s_exo_names = struct('x', 'coordx','y', 'coordy', 'z', 'coordz', 'basal_friction','log_basal_friction', 'ice_thickness', 'ice_thickness', 'surface_height', 'surface_height', 'bed_topography', 'bed_topography', 'T', 'temperature', 'global_vars','layer_thickness_ratio_01', 'vx', 'solution_1', 'vy', 'solution_2','vz','solution_3');

s_geo = exo_read( [fdir_in, exo_fname_in], s_exo_names);

%% extract fields from s_geo struct

exo_beta = s_geo.basal_friction(:,end);
exo_H = s_geo.ice_thickness(:,end);
exo_b = s_geo.bed_topography(:,end);
exo_h = s_geo.surface_height(:,end);
exo_T = s_geo.T(:,end);
exo_vx = s_geo.vx(:,end);
exo_vy = s_geo.vy(:,end);
exo_vz = s_geo.vz(:,end);

nLevels = length(s_geo.global_vars)-1
isAlbanyMeshOrderColumnWise = (s_geo.global_vars(end-1));

thick_levels = zeros(nLevels,1);
for i=1:nLevels-1
  thick_levels(i+1) = thick_levels(i) + s_geo.global_vars(i);
end


beta = exo_beta(1:nLevels:end);
H = exo_H(1:nLevels:end);
b = exo_b(1:nLevels:end);
h = exo_h(1:nLevels:end);


fout = fopen ( ['surface_height.ascii'], 'wt' );
fprintf(fout,'%d\n', length(beta));
fprintf(fout,'%.15g\n', h);
fclose(fout);
fout = fopen ( ['log_basal_friction.ascii'], 'wt' );
fprintf(fout,'%d\n', length(beta));
fprintf(fout,'%.15g\n', beta);
fclose(fout);
fout = fopen ( ['basal_friction.ascii'], 'wt' );
fprintf(fout,'%d\n', length(beta));
fprintf(fout,'%.15g\n', exp(beta));
fclose(fout);
fout = fopen ( ['thickness.ascii'], 'wt' );
fprintf(fout,'%d\n', length(beta));
fprintf(fout,'%.15g\n', H);
fclose(fout);
fout = fopen ( ['bed_topography.ascii'], 'wt' );
fprintf(fout,'%d\n', length(beta));
fprintf(fout,'%.15g\n', b);
fclose(fout);


T = reshape(exo_T, nLevels, length(exo_T)/nLevels);

%mean(T(11,:))
T = reshape(T', numel(T),1);
fout = fopen (['temperature.ascii'], 'wt' );
fprintf(fout,'%d %d\n', length(T)/nLevels, nLevels);
fprintf(fout,'%g\n', thick_levels);
fprintf(fout,'%g\n', T);
fclose(fout);


vz = reshape(exo_vz, nLevels, length(exo_vz)/nLevels);
vz = reshape(vz', numel(vz),1);
fout = fopen (['vertical_velocity.ascii'], 'wt' );
fprintf(fout,'%d %d\n', length(vz)/nLevels, nLevels);
fprintf(fout,'%g\n', thick_levels);
fprintf(fout,'%g\n', vz);
fclose(fout);


vx = reshape(exo_vx, nLevels, length(exo_vx)/nLevels);
%vx(2,1:3)
%vx = reshape(vx', numel(vx),1);
vy = reshape(exo_vy, nLevels, length(exo_vy)/nLevels);
%vy(1,1:3)
%vy = reshape(vy', numel(vy),1);
v = zeros(size(vx,2),2,nLevels);
v(:,1,:) = vx';
v(:,2,:) = vy';
%v(:,1:2:end) = vx;
%v(:, 2:2:end) = vy;
%v = reshape(v, numel(v),1

v = [];%zeros(2*numel(vx),1);
for il = 1:nLevels
    v = [v, vx(il,:), vy(il,:)];
end

%v = [vx;vy];
numComps = length(vz)/nLevels;
%for i=0:nLevels-1
%  v(2*i*numComps+1:(2*i+1)*numComps) = vx(i*numComps+1:(i+1)*numComps);
%  v((2*i+1)*numComps+1:(2*i+2)*numComps) = vy(i*numComps+1:(i+1)*numComps);
%end
fout = fopen (['horizontal_velocity.ascii'], 'wt' );
fprintf(fout,'%d %d %d\n', length(vz)/nLevels, 2, nLevels);
fprintf(fout,'%g\n', thick_levels);
fprintf(fout,'%g\n', v);
fclose(fout);



