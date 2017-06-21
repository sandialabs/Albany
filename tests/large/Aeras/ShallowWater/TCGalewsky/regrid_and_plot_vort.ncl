;======================================================================
; ESMF_regrid_10.ncl
;
; Concepts illustrated:
;   - Interpolating from one grid to another using ESMF_regrid
;   - Interpolating data from an MPAS grid to 0.25 degree grid
;======================================================================
; This example is identical to ESMF_all_10.ncl, except it does the
; regridding in one call to "ESMF_regrid".  See ESMF_wgts_10.ncl
; for a faster example of regridding using an existing weights file.
;======================================================================
; This example uses the ESMF application "ESMF_RegridWeightGen" to 
; generate the weights.
;
; For more information about ESMF:
;
;        http://www.earthsystemmodeling.org/
;
; This script uses built-in functions that will be in NCL version
; 6.1.0.
;======================================================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin
;---Retrieve data

   srcFileName = "TCGalewsky.nc";
   sfile = addfile(srcFileName,"r")
   f = sfile;

   numnodes=dimsizes(f->node_num_map);
   numtimes = dimsizes(f->time_whole)
   x=f->coordx;
   y=f->coordy;
   z=f->coordz;
   lon=new(numnodes,double);
   lat=new(numnodes,double);
   sval = csc2s(x,y,z);
   print("numnodes = "+numnodes+ ", numtimes = "+numtimes);

  lat = sval(0,:);
  lon = sval(1,:);

  uvel    = sfile->vals_nod_var2(numtimes-1,:)
  vvel    = sfile->vals_nod_var3(numtimes-1,:)

  Opt                 = True                     ; Regridding optioins

  Opt@SrcFileName     = "MPAS_ESMF.nc"           ; output files
  Opt@DstFileName     = "World_0.25deg_SCRIP.nc"
  Opt@WgtFileName     = "MPAS_2_World.nc"
  Opt@ForceOverwrite  = True

  Opt@SrcGridLat      = lat               ; source grid
  Opt@SrcGridLon      = lon
  Opt@InputFileName   = srcFileName              ; optional

  Opt@DstGridType     = "4x4"                ; destination grid
; Opt@DstTitle        = "World Grid 0.25 degree resolution"
; Opt@DstLLCorner     = (/-89.75d,   0.00d /)
; Opt@DstURCorner     = (/ 89.75d, 359.75d /) 

  Opt@Debug           = True
;;Opt@PrintTimings  = True

 ;   height_regrid = ESMF_regrid(height,Opt)
    uvel_regrid = ESMF_regrid(uvel,Opt)
    vvel_regrid = ESMF_regrid(vvel,Opt)    


   dims = dimsizes(uvel_regrid);
   dim1 = dims(0);
   dim2 = dims(1);
   
   vort = new( (/dim1, dim2/), double)

   uv2vrf(uvel_regrid, vvel_regrid, vort)

;----------------------------------------------------------------------
; Plotting section
;----------------------------------------------------------------------
    wks = gsn_open_wks("pdf","ESMF_regrid")
;    gsn_define_colormap(wks,"rainbow")             ; Change color map

;---Resources to share between both plots
    res                     = True              ; Plot modes desired.
;    res@gsnDraw             = False
;    res@gsnFrame            = False
    res@gsnMaximize         = True              ; Maximize plot
;    res@cnFillOn            = True              ; color plot desired
    res@cnLinesOn           = True             ; turn off contour lines
;    res@cnLineLabelsOn      = False             ; turn off contour labels
    res@cnFillMode          = "RasterFill"      ; turn raster on      
    res@lbLabelBarOn        = False    ; Will turn on in panel later.
    res@mpFillOn            = False
    res@mpOutlineOn = False;


;---VORTICITY
  res@cnLevelSelectionMode = "ManualLevels";
  res@cnMinLevelValF = min(vort);
  res@cnMaxLevelValF = max(vort);
  res@cnLevelSpacingF = 0.00002;
  
  print("max = "+max(vort)+",  min="+min(vort))

  res@tiMainString          = "vorticity"
  map = gsn_csm_contour_map(wks,vort,res)

end