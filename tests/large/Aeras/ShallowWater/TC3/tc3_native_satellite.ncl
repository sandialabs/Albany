load "~/local/ncl/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "~/local/ncl/lib/ncarg/nclscripts/csm/gsn_csm.ncl"


;;; Plotting TC3 on native grid, satellite view
;;; Note that *exo file should be renamed to *nc file

begin



 f=addfile("TC3.nc","r");

 numnodes=dimsizes(f->node_num_map);

 numtimes = dimsizes(f->time_whole)

 x=f->coordx;
 y=f->coordy;
 z=f->coordz;
 lon=new(numnodes,double);
 lat=new(numnodes,double);

 ;;get lon/lat of nodes
 sval = csc2s(x,y,z);

 lat = sval(0,:);
 lon = sval(1,:);

 print("how many time frames = "+numtimes)

 ;;; ...var1 = geopotential, ...var2 = U, ...var3 = V
 outdata_max_error = f->vals_nod_var1(numtimes - 1,:) - f->vals_nod_var1(0,:);

 print("max abs error = " + max(abs(outdata_max_error)))

 outdata = f->vals_nod_var1(0,:) ;

;******************************************************
; create plot
;******************************************************

 wks = gsn_open_wks("pdf","TC3_satellite_view") ; open workstation
 gsn_define_colormap(wks,"BlAqGrYeOrReVi200"); choose colormap

 res = True ; plot modifications desired
 ;res@gsnMaximize = True ; Maximize size of plot in frame

 res@sfXArray = lon  ;Required to tell NCL where to
 res@sfYArray = lat  ;overlay data on globe.
  
 res@cnFillOn = True ;Turn on contour fill

 res@cnLinesOn=False; Turn off contour lines
 res@cnLineLabelsOn=False; Turn off contour line labels
 res@lbLabelAutoStride=True; Clean up labelbar labels.

 ;res@cnLevelSelectionMode = "ManualLevels";

 res@mpProjection = "satellite";
 res@mpSatelliteDistF=3.0;
 res@mpCenterLonF               = -120.0     ; choose center lon
 res@mpCenterLatF               = 0.         ; choose center lat
 map = gsn_csm_contour_map_ce(wks,outdata,res)

end
