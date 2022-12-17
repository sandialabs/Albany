
s_exo_names = struct('z', 'coordz', 'T','Temperature', 'phi' ,'phi',  'E', 'solution');
s = exo_read('./kleiner_B.exo', s_exo_names);

dim = length(s.z)/4;
p.z = s.z(1:dim)/0.2;
rho = 910;
g = 9.81;
gamma = 4/180*pi;
A = 5.3e-24;
H = 200;
secperyr = 3.1556926e7;

coeff = A * (rho*g*sin(gamma))^3 /2;
vel_x = coeff * (H^4 - (H-H*p.z').^4) * secperyr;
vel = zeros(1, 2*dim);
vel(1:2:end) = vel_x;

vel = kron(vel,ones(1,4));


% fout = fopen ( ['./meshB/vel_xy.ascii'], 'wt' );
% fprintf(fout,'%d %d %d\n', 4, 2, dim);
% fprintf(fout,'%g\n', p.z);
% fprintf(fout,'%g\n', vel);
% fclose(fout);



p.E = (s.E(1:dim)+s.E(dim+1:2*dim)+s.E(2*dim+1:3*dim)+s.E(3*dim+1:4*dim))/4*1e3/910;
p.phi = (s.phi(1:dim)+s.phi(dim+1:2*dim)+s.phi(2*dim+1:3*dim)+s.phi(3*dim+1:4*dim))/4;
p.T = (s.T(1:dim)+s.T(dim+1:2*dim)+s.T(2*dim+1:3*dim)+s.T(3*dim+1:4*dim))/4-273.15;
plot(p.E,p.z);
p.layers = dim-1;
p.diffLength = 2.5/20;
p.name = ['Layers',num2str(p.layers),'_DiffLength',num2str(p.diffLength)];

load enthB_analy_result.mat
hold on
plot(enthB_analy_E/1000, enthB_analy_z)

%save([p.name,'.mat'],'p');
