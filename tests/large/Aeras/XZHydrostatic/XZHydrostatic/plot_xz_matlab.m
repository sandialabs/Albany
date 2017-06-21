%%%-----------------------------------------------------------
%%% Script to make contour plots of quantities remapped to z-surfaces
%%% from *exo file for XZHydrostatic.
%%% Matlab interface tools MEXNC ( url ) , etc. can be used 
%%% as well as native Matlab functions. 
%%% Note that MEXNC tools read vars in a native to netcdf format order,
%%% that is, if ncdump shows var1(times, lats, lons), then this is what
%%% nc_varget will read. Native Matlab functions reverse the order, so that
%%% netcdf.getVar returns var1(lons,lats,times).

function main_matlab_only

fil = '~/albany/Albany-rep/build/examples/Aeras/XZHydrostatic/xzhydrostatic.exo';

ps_file = getNCVar(fil, 'vals_nod_var1');
coordx = getNCVar(fil, 'coordx');
dims = size(ps_file);
numtimes = dims(2);
numnodes = dims(1);

lasttime = numtimes;

% Get name and length of last dimension
% Note that index 15 (0-based) is from running ncdump
ncid = netcdf.open(fil,'NC_NOWRITE');
[dimname, numnodvar] = netcdf.inqDim(ncid,15);
netcdf.close(ncid);
%%%%
%%%%%%%%%%%%%%%% how many levels: varis are ps, u, T, q0, q1 ,...
%%%%%%% where is dp/deta????

%POTENTIAL PROBLEM -- no number of levels or # of tracers
% in nc file
numlev = 30;
numtrac = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% global vars
global  gravity R ptop pbottom etatop aprime bprime;
pbottom = 101325.0;
ptop = 101.325;
etatop = ptop/pbottom;
gravity = 9.87;
R = 287.058;

ninter = numlev + 1;

step = (etatop - 1)/(numlev);
etainter = [1:step:etatop];
etam = (etainter(1:end-1)+etainter(2:end))/2;
aprime = etatop/(etatop-1);
bprime = 1/(1-etatop);


ps = zeros(1,numnodes);
ps(:) = ps_file(:,lasttime);
p = zeros(numlev, numnodes);
dpdeta = p;
T = p;
u = p;
F = p;
zsurf = p;
q0 = p;

figure(1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%create arrays p, dp/deta
for nn = 1:numlev
   ee = etam(nn);
   p(nn,:) = fun_p(ps, ee);
   dpdeta(nn,:) = fun_dpdeta(ps);
   plot(coordx, p(nn,:), '-*'); hold on;
end

%%%%%%%%%%% now we read u,T from file
pref = 'vals_nod_var';
levnumb = 1;
for lev=1:numlev
    levnumb = levnumb + 1;
    suff = num2str(levnumb);
    namevar = strcat(pref,suff);
    var_file = getNCVar(fil, namevar);
    u(lev,:) = var_file(:,lasttime);
    levnumb = levnumb + 1;
    suff = num2str(levnumb);
    namevar = strcat(pref,suff);
    var_file = getNCVar(fil, namevar);
    T(lev,:) = var_file(:,lasttime);   
end

%%%%%%%%%%% now we read q0 from file
%levnumb is assigned above
for lev=1:numlev
    levnumb = levnumb + 1;
    suff = num2str(levnumb);
    namevar = strcat(pref,suff);
    var_file = getNCVar(fil, namevar);
    q0(lev,:) = var_file(:,lasttime);   
end

%%%%%%%%% other tracers can be done in the same way...

%%%%%%%%%%%%%%%%%%%%%%%% now let's get F...
%note that Fs is not in the file either. It can be given by 
%an analyt function but one should verify it is the same in here!

Fs = ps; 
Fs(:) = 0;

for j=1:numlev
   ee = etam(j);
   for k=1:numnodes
       fs = Fs(k);
       ff = 0;
       for sublev=1:j-1
           ff = ff - (R*T(sublev,k)./p(sublev,k)...
                .*dpdeta(sublev,k)*step)';
       end       
       ff = ff - (R*T(j,k)./p(j,k).*...
                dpdeta(j,k)*step/2)';
       
       F(j,k) = ff + fs;
   end
end

zsurf = F/gravity;

figure(2)
for j=1:numlev
    plot(coordx,zsurf(j,:),'*-'); hold on;
end


plotcont(u, etam, coordx, zsurf, 3, 'velocity')


aa = 1;
    





%%%%%%%%%%%% FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res = a(eta)
res = eta;
res = eta - b(eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res = b(eta)
global etatop
res = eta;
res = (eta - etatop)/(1-etatop);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res = fun_dpdeta(ps)
global pbottom aprime bprime;
res = ps; res(:) = 0;

res = ps*bprime+pbottom*aprime;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%pressure of x and eta, assume that ps is an array, eta is scalar
function res = fun_p(ps,eta)
global pbottom;
res = ps; res(:) = 0;

res = ps*b(eta)+pbottom*a(eta);


%%%%%%%% plotting contours

function plotcont(field, etam, xpos, zsurf, fign, figtitle)

nlev = length(etam);
npts = length(xpos);

x = []; z = []; data = []; 
for j=1:nlev
    ee = etam(j);
    for k=1:npts
        xx=xpos(k);
        zz=zsurf(j,k);
        x = [x xx];
        z = [z zz];
        data = [data field(j,k)];
    end
end

zmax = max(zsurf(nlev,:));

xlin=linspace(min(xpos),max(xpos),npts);
zlin=linspace(0,zmax,2*nlev);
[X,Y]=meshgrid(xlin,zlin);
Z=griddata(x,z,data,X,Y,'cubic');
%
figure(fign)
%contour(X,Y,Z, 'ShowText', 'on'); % interpolated
contourf(X,Y,Z); % interpolated
colorbar
title(figtitle)


function data = getNCVar(filename, varname)
ncid = netcdf.open(filename,'NC_NOWRITE');

% Get the name of the first variable.
%[varname, xtype, varDimIDs, varAtts] = netcdf.inqVar(ncid,0);
% Get variable ID of the first variable, given its name.
varid = netcdf.inqVarID(ncid,varname);
% Get the value of the first variable, given its ID.
data = netcdf.getVar(ncid,varid);
netcdf.close(ncid);

