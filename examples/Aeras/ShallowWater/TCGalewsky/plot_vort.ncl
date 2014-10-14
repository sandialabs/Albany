load "~/local/ncl/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "~/local/ncl/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin

  ffile = "TCGalewsky.nc";
  f = addfile(ffile,"r");

  ds = dimsizes(f->variable_0);
  lasttime = ds(0)-1;
  
  uvel = f->variable_1(lasttime,0,:,:) ;
  vvel = f->variable_1(lasttime,0,:,:) ; 
  
  vort = uvel ; this way attrs are copied too

  uv2vrf(uvel, vvel, vort)
   
;******************************************************
; create plot
;******************************************************

wks = gsn_open_wks("pdf","vort_interp") ; open workstation
gsn_define_colormap(wks,"BlAqGrYeOrReVi200"); choose colormap

res = True ; plot modifications desired

  
  res@cnFillOn=True;Turn on contour fill
;  res@cnFillMode  = "AreaFill"  ; Style of fill. You can also
; use "CellFill" and "RasterFill"

  res@cnLinesOn=False; Turn off contour lines
  res@cnLineLabelsOn=False; Turn off contour line labels
  res@lbLabelAutoStride=True; Clean up labelbar labels.

; res@tiMainString = names(0); Title

  mincont = min(vort);
  maxcont = max(vort);
 
  numcont = 100;

  res@cnLevelSelectionMode = "ManualLevels";
  res@cnMinLevelValF = mincont;
  res@cnMaxLevelValF = maxcont
  res@cnLevelSpacingF = (maxcont - mincont)/numcont;
  
  print("field's min, max = "+ mincont+"   "+ maxcont)

  res@mpOutlineOn = False;
  res@mpFillOn = False;

  map = gsn_csm_contour_map_ce(wks,vort,res)
end
