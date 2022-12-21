H = 1.06; % km
L = 50;   % km
W = 10;   % km

nex = 50;
ney = 10;

npx = nex + 1;
npy = ney;  % periodic in y, so one less point

np = npx*npy;

x = linspace(0,L,npx);
H_min = min(0.001,H/npx); % make sure there are no flat elements at the end

% Geometry (ice thickness and surface height)
func_thickness = @(x)(max(H_min,H*sqrt(1-x/L)));
s = func_thickness(x);

surface_height = [];
for i=1:npy
  surface_height = [surface_height; s'];
end
ice_thickness = surface_height;

sh_fid = fopen("surface_height.ascii",'w');
th_fid = fopen("ice_thickness.ascii",'w');

fprintf(sh_fid,'%d\n',np);
fprintf(th_fid,'%d\n',np);

fprintf(sh_fid,'%f\n',surface_height);
fprintf(th_fid,'%f\n',ice_thickness);

fclose(sh_fid);
fclose(th_fid);

% Hydrology (surface water input)

rm = 25;  % mm/day
rs = 60;  % mm/(day*km)
sm = 0.5; % km
func_water_input = @(x)(max(0,rm-rs*abs(s-sm)));
swi = func_water_input(x);

surface_water_input = [];
for i=1:npy
  surface_water_input = [surface_water_input; swi'];
end

swi_fid = fopen("surface_water_input.ascii",'w');

fprintf(swi_fid, '%d\n', np);

fprintf(swi_fid, '%f\n', surface_water_input);

fclose(swi_fid);
