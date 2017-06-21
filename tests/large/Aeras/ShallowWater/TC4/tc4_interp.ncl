load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

; plotting TC4 numerical solution on interpolated grid
begin

myPi = 3.1415926;

f = addfile("TC4.nc", "r");

ds = dimsizes(f->variable_0);
lasttime = ds(0)-1;

var1 = f->variable_0(lasttime,0,:,:);
var2 = f->variable_1(lasttime,0,:,:);
var3 = f->variable_2(lasttime,0,:,:);

wks = gsn_open_wks("pdf","tc4_numeric")
gsn_define_colormap(wks,"rainbow")             ; Change color map
res                     = True              ; Plot modes desired.
res@cnFillOn            = True              ; color plot desired
res@cnLinesOn           = False            ; turn off contour lines

;move the translating low to the middle of the plot
res@mpCenterLonF = 180.

map = gsn_csm_contour_map(wks,var1,res)
     
end
  
  
  
  
  
  
  
  
  