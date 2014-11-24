
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

function dbubf(lat:double)
local rum, coslat, output
begin  
 su0 = 20;
 npwr = 14;
  rmu = sin(lat);
  coslat = cos(lat);
  output = 2.*su0*(2.*rmu*coslat)^(npwr-1.)*(npwr-(2.*npwr+1)*rmu*rmu);
  return(output);
end
function bubfnc(lat:double)
local output
begin
 su0 = 20;
 npwr = 14;
 output = su0*(2.*sin(lat)*cos(lat))^npwr;
 return(output)
end
function phicon(lat:double)
local a, int_steps, deltat, h, myPi, midpoint1, midpoint2, loc_u, Omega
begin
  myPi = 3.1415926;
  a = 6.37122e6;
  Omega = 2.0*myPi/(24.*3600.);
  int_steps = 100;
  
  h =0.d;
  
  deltat = (lat+myPi/2.0)/int_steps;
  do i=0,int_steps-1
    
    midpoint1 = -myPi/2.0 + (i-1)*deltat;
    midpoint2 = -myPi/2.0 + i*deltat;
    
    loc_u = bubfnc(midpoint1);
    h = h - a*deltat*(2*Omega*sin(midpoint1)+loc_u*tan(midpoint1)/a)*loc_u/2.;
    
    loc_u = bubfnc(midpoint2);
    
    h = h- a*deltat*(2*Omega*sin(midpoint2)+loc_u*tan(midpoint2)/a)*loc_u/2.;
    
  end do

  h = (1.0e5 + h)/9.80616;
  
  return(h);
  
end



begin

myPi = 3.1415926;

f = addfile("256_128grid.nc", "r");

lon = (f->lon)*myPi/180.;
lat = (f->lat)*myPi/180;
var1 = f->variable_0(0,0,:,:);
var2 = f->variable_0(0,0,:,:);
var3 = f->variable_0(0,0,:,:);

size = dimsizes(lon);
lonsize = size(0);

size = dimsizes(lat);
latsize = size(0);

 
 
  
  earthRadius = 6.37122e6;
  su0 = 20.d;
  phi0 = 1.0e5;
  rlon0 = 0.d;
  rlat0 = myPi/4.d;
  npwr = 14.d;
  
  Omega = 2.0*myPi/(24.*3600.); //this should be sitting in SW class
  gravity = 9.80616
  
  a = earthRadius;
    
  tol = 1.e-10;
  
  alfa = -0.03*(phi0/(2.*Omega*sin(myPi/4.)));
  sigma = (2.*a/1.0e6)*(2.*a/1.0e6);

  ai = 1./a;
  a2i = ai*ai;

;;;;;;;;;;;;;;;;;;;;;NOTE THAT YOU NEED TO SET A DAY
day = 3;
time = day*86400;
tmshft = su0*time/a;

 var1(:,:) = 0.d;


do i=0,lonsize-1
do j=0,latsize-1

  lambda = lon(i);
  theta = lat(j);
  
 ; print("lon lat "+lambda+" "+theta)

   
  

  
   snj = sin(theta);
   csj = cos(theta)*cos(theta);
   srcsj = cos(theta);
   tmpry = tan(theta);
   tmpry2 = tmpry*tmpry;
   den = 1./cos(theta);
   aacsji = 1./(a*a*csj);
   corr = 2.*Omega*snj;
  
  
   ucon = bubfnc(theta);
   bigubr = ucon*srcsj; ///
   dbub = dbubf(theta); ///
  
   c = sin(rlat0)*snj + cos(rlat0)*srcsj*cos(lambda - tmshft - rlon0);
  

   psib = 0.d;

  if( abs(c+1.) .gt. tol )then
     psib = alfa*exp(-sigma*((1.-c)/(1.+c)));
  end if
  
   dcdm = sin(rlat0)-cos(lambda-tmshft -rlon0)*cos(rlat0)*tmpry;
   dcdl = -cos(rlat0)*srcsj*sin(lambda-tmshft-rlon0);
   d2cdm = -cos(rlat0)*cos(lambda-tmshft-rlon0)*(1.+tmpry2)/srcsj;
   d2cdl = -cos(rlat0)*srcsj*cos(lambda-tmshft-rlon0);
  
   tmp1 = 0.d;
  if( abs(c+1.) .ge. tol ) then
    tmp1 = 2.*sigma*psib/((1.+c)*(1.+c));
   end if
   tmp2 = 0.d;
  if( abs(c+1.) .ge. tol) then
    tmp2 = (sigma - (1.0+c))/((1.+c)*(1.+c));
    end if
    
   dkdm = tmp1*dcdm;
   dkdl = tmp1*dcdl;
  
   d2kdm  = tmp1*(d2cdm + 2.0*(dcdm*dcdm)*tmp2);
   d2kdl  = tmp1*(d2cdl + 2.0*(dcdl*dcdl)*tmp2);
  
  u = bigubr*den - srcsj*ai*dkdm;
  v = (dkdl*ai)*den;
  
  ;comment var1 out for speed if var1 is not plotted
  var1(j,i) = phicon(theta) +corr*psib/gravity;
  var2(j,i) = u;
  var3(j,i) = v;

  
  end do
  end do

   
  wks = gsn_open_wks("pdf","tc4_an")
  gsn_define_colormap(wks,"rainbow")             ; Change color map

;---Resources to share between both plots
  res                     = True              ; Plot modes desired.
    
  res@cnFillOn            = True              ; color plot desired
  res@cnLinesOn           = False            ; turn off contour lines

  res@tiMainString          = "vorticity"
  map = gsn_csm_contour_map(wks,var1(:,:),res)
    
  end
  
  
  
  
  
  
  
  
  