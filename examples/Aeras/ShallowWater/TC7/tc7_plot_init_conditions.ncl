load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin

;;;; init conditions and T=day1 fields downloaded from
;;;; http://www.csm.ornl.gov/chammp/stswm/ for TC7

fref = addfile("REF0007.cdf", "r");

gravity = 9.80616d
R = 6.37122e6;

;; plotting T=0 or T=1 (day)
time = 0;
;; which variable? "geop" for geopotential or "vel" for velocities
varname = "vel"

;; resolution of interpolated fields
nlat = 200
nlon = 400

;;;;;;;;; this part of the code is 
;;;; borrowed from HOMME TC5 plotting routine
nlat@double = True
lat  = latGau(nlat, "lat", "latitude", "degrees_north")
gwt  = latGauWgt(nlat, "lat", "gaussian weights", "")
lon  = lonGlobeF(nlon, "lon", "longitude", "degrees_east")

; Convert NCAR geop reference solution to lat-lon
tmp = dimsizes(fref->geopotential)
ntimes=tmp(0)
nwave=tmp(1)
mwave=tmp(2)

if( varname .eq. "geop")then
;;;this part of the code is taken from HOMME's script
;;;for TC5, from obtaining a ref solution from the spectral core 
  geohatr = new ( (/nlat,nlat/), double)
  geohati = new ( (/nlat,nlat/), double)
  geo_ref = new ( (/nlat,nlon/), double)
  geo_ref!0="lat"
  geo_ref!1="lon"
  geo_ref&lat = lat    
  geo_ref&lon = lon
  geohatr = 0
  geohati = 0

  geohatr(0:nwave-1,0:mwave-1) = 2*fref->geopotential(time,:,:,0)
  geohati(0:nwave-1,0:mwave-1) = 2*fref->geopotential(time,:,:,1)
  shsgc(geohatr,geohati,geo_ref)
;;;end of borrowed code
  geo_ref = geo_ref/gravity
end if

if ( varname .eq. "vel" ) then
  vorthatr = new ( (/nlat,nlat/), double)
  vorthati = new ( (/nlat,nlat/), double)
  vort_ref = new ( (/nlat,nlon/), double)
  vort_ref!0="lat"
  vort_ref!1="lon"
  vort_ref&lat = lat    
  vort_ref&lon = lon
  vorthatr = 0
  vorthati = 0
  divhatr = vorthatr;
  divhati = vorthati;
  div_ref = vort_ref;

  vorthatr(0:nwave-1,0:mwave-1) = 2*fref->vorticity(time,:,:,0)
  vorthati(0:nwave-1,0:mwave-1) = 2*fref->vorticity(time,:,:,1)
  shsgc(vorthatr,vorthati,vort_ref)
 
  divhatr(0:nwave-1,0:mwave-1) = 2*fref->divergence(time,:,:,0)
  divhati(0:nwave-1,0:mwave-1) = 2*fref->divergence(time,:,:,1)
  ;;;; transform spectral coefs to grid values
  shsgc(divhatr,divhati,div_ref)

  ;get velocities from vorticity/divergence
  ucomp = vort_ref;
  vcomp = vort_ref;
  ucomp = 0
  vcomp = 0
  vrdv2uvg(vort_ref, div_ref, ucomp, vcomp)

;;lets do a sanity check, transform velocities to
;;vorticity/divergence and see if there is any difference?
;;result: there is no difference.
;  vv = ucomp;
;  dd = ucomp;
;  vv = 0
;  dd = 0 
;  uv2vrdvg(ucomp, vcomp, vv, dd)

 ucomp_R = ucomp;
 vcomp_R = vcomp;

 ucomp_R = ucomp/R;
 vcomp_R = vcomp/R;

end if



filename = "swtc7time"+time+"_"+varname;

if ( varname .eq. "geop" )then 

  wks = gsn_open_wks("pdf",filename)
  res=True

  ;res@tiMainString        = "NCAR T213 Reference (geop)"

  gsn_define_colormap(wks,"rainbow")             ; Change color map
  res@cnFillOn            = True              ; color plot desired
  res@cnLinesOn           = False            ; turn off contour lines
  contour = gsn_csm_contour_map(wks,geo_ref,res)     ; Draw a contour plot.  ;
end if

if ( varname .eq. "vel" )  then 
  wks = gsn_open_wks("pdf",filename)
  res=True

  ;;;res@tiMainString        = "NCAR T213 Reference (geop)"

  gsn_define_colormap(wks,"rainbow")             ; Change color map
  res@cnFillOn            = True              ; color plot desired
  res@cnLinesOn           = False            ; turn off contour lines
  contour = gsn_csm_contour_map(wks,ucomp_R,res)     ; Draw a contour plot.  ;
  contour = gsn_csm_contour_map(wks,vcomp_R,res)     ; Draw a contour plot.  ;  
  
  ;;;;; part of the sanity check, plotting
  ;wks2 = gsn_open_wks("pdf","tc7_vort_diff")
  ;res=True
  ;gsn_define_colormap(wks2,"rainbow")             ; Change color map
  ;res@cnFillOn            = True              ; color plot desired
  ;res@cnLinesOn           = False            ; turn off contour lines
  ;diff1 = vv ; to copy attributes
  ;diff1 = vv-vort_ref
  ;contour = gsn_csm_contour_map(wks2,diff1,res)     ; Draw a contour plot.  ;
  ;contour = gsn_csm_contour_map(wks2,vort_ref,res)     ; Draw a contour plot.  ;   
end if


end