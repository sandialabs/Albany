load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin

;;;; init conditions and T=day1 fields downloaded from
;;;; http://www.csm.ornl.gov/chammp/stswm/ for TC7

fref = addfile("REF0007.cdf", "r");
fic = addfile("grid_and_IC_16_quad9.nc","w")

gravity = 9.80616d
R = 6.37122e6;
time = 0;


;; resolution of interpolated fields
;; G64 grid
nlat = 64
nlon = 128

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


;;;;;;;;;;;;;;;;;;;; gauss to .25 degree grid

;jlat=73,ilon=72
;X = g2fsh (x, (/jlat,ilon/))


;;;;;now we need to read unstructured grid







;---Output (and input) files for regridding
    srcGridName = "structured.nc"
    dstGridName = "unstructured.nc"
    wgtFile     = "weights.nc"

;---Set to True if you want to skip any of these steps
    SKIP_ESMF_GEN  = False
    SKIP_SCRIP_GEN = False
    SKIP_WGT_GEN   = False
    
;----------------------------------------------------------------------
; Convert Aeras to unstructured ESMF file.
;----------------------------------------------------------------------

    numnodes=dimsizes(fic->node_num_map);
    x=fic->coordx;
    y=fic->coordy;
    z=fic->coordz;
    lonic=new(numnodes,double);
    latic=new(numnodes,double);

    sval = csc2s(x,y,z);
    
   ;;;; in degrees
    latic = sval(0,:);
    lonic = sval(1,:);
    
    do i=1,numnodes - 1
       if ( lonic(i) .ge. 180.0 ) then
            lonic(i) = lonic(i) - 360.0;
        end if
    end do
 
    lat2 = latic;
    lon2 = lonic;


;---Convert to degrees from radians
    r2d     = 180.0d/(atan(1)*4.0d)  ; Radian to Degree
    ;lonic = lonic*r2d
    ;latic = latic*r2d

    if(.not.SKIP_ESMF_GEN) then
;---Set some Options
      Opt                = True
      Opt@ForceOverwrite = True
      Opt@PrintTimings   = True
;     Opt@InputFileName  = srcFileName
      Opt@Debug          = True

      unstructured_to_ESMF(dstGridName,latic,lonic,Opt)

;---Clean up
      delete(Opt)
    end if

;----------------------------------------------------------------------
; Convert G64 world grid to SCRIP file
;----------------------------------------------------------------------
    if(.not.SKIP_SCRIP_GEN) then
      Opt                = True
      ;Opt@LLCorner       = (/-89.75d,   0.00d /)
      ;Opt@URCorner       = (/ 89.75d, 359.75d /) 
      Opt@ForceOverwrite = True
      Opt@PrintTimings   = True
      Opt@Title          = "World Grid G128 degree resolution"
      Opt@Debug          = True

      latlon_to_SCRIP(srcGridName,"G128",Opt)

;---Clean up
      delete(Opt)
    end if

;----------------------------------------------------------------------
; Generate interpolation weights for World Grid to Aeras grid    
;----------------------------------------------------------------------
    if(.not.SKIP_WGT_GEN) then
      Opt                = True
      Opt@InterpMethod   = "bilinear"
      Opt@DstESMF        = True    
      Opt@ForceOverwrite = True
      Opt@PrintTimings   = True

      ESMF_regrid_gen_weights(srcGridName, dstGridName, wgtFile, Opt)

;---Clean up
      delete(Opt)
    end if

;----------------------------------------------------------------------
; Interpolate data
;----------------------------------------------------------------------

;---ger grop interpolated data
    sp = geo_ref
    Opt                = True
    Opt@Debug          = True
    Opt@PrintTimings   = True
    Opt@Debug          = True
;---regrid
    sp_regrid = ESMF_regrid_with_weights(sp,wgtFile,Opt)

    fic->vals_nod_var1(0,:) = sp_regrid(:)
    
;---get velocities interpolated    
;--- DO NOT FORGET R = EARTH RADIUS,
; since winds calculations from div/vort seem to consider a unit sphere
    sp = ucomp
    sp_regrid = ESMF_regrid_with_weights(sp,wgtFile,Opt)
    fic->vals_nod_var2(0,:) = sp_regrid(:)/R
    sp = vcomp
    sp_regrid = ESMF_regrid_with_weights(sp,wgtFile,Opt)
    fic->vals_nod_var3(0,:) = sp_regrid(:)/R







wks = gsn_open_wks("pdf","var_remapped") ; open workstation
gsn_define_colormap(wks,"rainbow"); choose colormap

res = True ; plot modifications desired

;;has to be in grads
 res@sfXArray= lon2;  Required to tell NCL where to
 res@sfYArray= lat2;   overlay data on globe.
  
  res@cnFillOn=True;Turn on contour fill

res@cnLinesOn=False; Turn off contour lines
res@cnLineLabelsOn=False; Turn off contour line labels
res@lbLabelAutoStride=True; Clean up labelbar labels.

var1 = fic->vals_nod_var3(0,:)
  map = gsn_csm_contour_map_ce(wks,var1,res)



end