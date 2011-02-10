save all
disp_x = solution_x
disp_y = solution_y
disp_z = solution_z
Fp_xx = state_01
Fp_xy = state_02
Fp_xz = state_03
Fp_yx = state_04
Fp_yy = state_05
Fp_yz = state_06
Fp_zx = state_07
Fp_zy = state_08
Fp_zz = state_09
eqps = state_10
stress_xx = state_11
stress_xy = state_12
stress_xz = state_13
stress_yx = state_14
stress_yy = state_15
stress_yz = state_16
stress_zx = state_17
stress_zy = state_18
stress_zz = state_19
vm=(1/sqrt(2))*TMAG(stress_xx,stress_yy,stress_zz,stress_xy,stress_yz,stress_zx)
pressure = (1/3)*(stress_xx + stress_yy + stress_zz)
delete solution_x, solution_y, solution_z 
delete state_01, state_02, state_03, state_04, state_05, state_06
delete state_07, state_08, state_09, state_10, state_11, state_12
delete state_13, state_14, state_15, state_16, state_17, state_18, state_19
exit