load "~/local/ncl/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "~/local/ncl/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

;;; Plots fields or error plots on native grids 
begin


f=addfile("TC3.nc","r");

numnodes=dimsizes(f->node_num_map);

numtimes = dimsizes(f->time_whole)

x=f->coordx;
y=f->coordy;
z=f->coordz;
lon=new(numnodes,double);
lat=new(numnodes,double);

sval = csc2s(x,y,z);

lat = sval(0,:);
lon = sval(1,:);

print("how many time frames = "+numtimes)


outdata_max_error = f->vals_nod_var1(numtimes - 1,:) - f->vals_nod_var1(0,:);

print("max abs error = " + max(abs(outdata_max_error)))

outdata = f->vals_nod_var1(0,:) ;


;******************************************************
; create plot
;******************************************************


wks = gsn_open_wks("pdf","tc3") ; open workstation
gsn_define_colormap(wks,"BlAqGrYeOrReVi200"); choose colormap

res = True ; plot modifications desired

res@sfXArray= lon;  Required to tell NCL where to
res@sfYArray= lat  ;overlay data on globe.
  
res@cnFillOn=True;Turn on contour fill

res@cnLinesOn=False; Turn off contour lines
res@cnLineLabelsOn=False; Turn off contour line labels
res@lbLabelAutoStride=True; Clean up labelbar labels.

res@tiMainString = "TC3"; Title

; res@cnLevelSelectionMode = "ManualLevels";

map = gsn_csm_contour_map_ce(wks,outdata,res)

end
