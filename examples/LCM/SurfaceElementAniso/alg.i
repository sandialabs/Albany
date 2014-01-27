save all

disp_x = solution_X
disp_y = solution_Y
disp_z = solution_Z

fint_x = residual_X
fint_y = residual_Y 
fint_z = residual_Z

stress_xx = (1/8)*(CAUCHY_STRESS_01+CAUCHY_STRESS_10+CAUCHY_STRESS_19+CAUCHY_STRESS_28+CAUCHY_STRESS_37+CAUCHY_STRESS_46 >
              +CAUCHY_STRESS_55+CAUCHY_STRESS_64)
stress_xy = (1/8)*(CAUCHY_STRESS_02+CAUCHY_STRESS_11+CAUCHY_STRESS_20+CAUCHY_STRESS_29+CAUCHY_STRESS_38+CAUCHY_STRESS_47 >
              +CAUCHY_STRESS_56+CAUCHY_STRESS_65)
stress_xz = (1/8)*(CAUCHY_STRESS_03+CAUCHY_STRESS_12+CAUCHY_STRESS_21+CAUCHY_STRESS_30+CAUCHY_STRESS_39+CAUCHY_STRESS_48 >
              +CAUCHY_STRESS_57+CAUCHY_STRESS_66)
stress_yy = (1/8)*(CAUCHY_STRESS_05+CAUCHY_STRESS_14+CAUCHY_STRESS_23+CAUCHY_STRESS_32+CAUCHY_STRESS_41+CAUCHY_STRESS_50 >
              +CAUCHY_STRESS_59+CAUCHY_STRESS_68)
stress_yz = (1/8)*(CAUCHY_STRESS_06+CAUCHY_STRESS_15+CAUCHY_STRESS_24+CAUCHY_STRESS_33+CAUCHY_STRESS_42+CAUCHY_STRESS_51 >
              +CAUCHY_STRESS_60+CAUCHY_STRESS_69)
stress_zz = (1/8)*(CAUCHY_STRESS_09+CAUCHY_STRESS_18+CAUCHY_STRESS_27+CAUCHY_STRESS_36+CAUCHY_STRESS_45+CAUCHY_STRESS_54 >
              +CAUCHY_STRESS_63+CAUCHY_STRESS_72)

surface_xx_1 = surface_cauchy_stress_01
surface_xy_1 = surface_cauchy_stress_02
surface_xz_1 = surface_cauchy_stress_03
surface_yy_1 = surface_cauchy_stress_05
surface_yz_1 = surface_cauchy_stress_06
surface_zz_1 = surface_cauchy_stress_09
surface_xx_2 = surface_cauchy_stress_10
surface_xy_2 = surface_cauchy_stress_11
surface_xz_2 = surface_cauchy_stress_12
surface_yy_2 = surface_cauchy_stress_14
surface_yz_2 = surface_cauchy_stress_15
surface_zz_2 = surface_cauchy_stress_18
surface_xx_3 = surface_cauchy_stress_19
surface_xy_3 = surface_cauchy_stress_20
surface_xz_3 = surface_cauchy_stress_21
surface_yy_3 = surface_cauchy_stress_23
surface_yz_3 = surface_cauchy_stress_24
surface_zz_3 = surface_cauchy_stress_27
surface_xx_4 = surface_cauchy_stress_28
surface_xy_4 = surface_cauchy_stress_29
surface_xz_4 = surface_cauchy_stress_30
surface_yy_4 = surface_cauchy_stress_32
surface_yz_4 = surface_cauchy_stress_33
surface_zz_4 = surface_cauchy_stress_36

surface_xx = (1/4)*(surface_xx_1 + surface_xx_2 + surface_xx_3 + surface_xx_4)
surface_xy = (1/4)*(surface_xy_1 + surface_xy_2 + surface_xy_3 + surface_xy_4)
surface_xz = (1/4)*(surface_xz_1 + surface_xz_2 + surface_xz_3 + surface_xz_4)
surface_yy = (1/4)*(surface_yy_1 + surface_yy_2 + surface_yy_3 + surface_yy_4)
surface_yz = (1/4)*(surface_yz_1 + surface_yz_2 + surface_yz_3 + surface_yz_4)
surface_zz = (1/4)*(surface_zz_1 + surface_zz_2 + surface_zz_3 + surface_zz_4)

vm = (1/sqrt(2))*TMAG(stress_xx,stress_yy,stress_zz,stress_xy,stress_yz,stress_xz)

pressure = (1/3)*(stress_xx + stress_yy + stress_zz)

delete solution_X, solution_Y, solution_Z 
delete residual_X, residual_Y, residual_Z

delete CAUCHY_STRESS_01, CAUCHY_STRESS_10, CAUCHY_STRESS_19, CAUCHY_STRESS_28, CAUCHY_STRESS_37, CAUCHY_STRESS_46, > 
       CAUCHY_STRESS_55, CAUCHY_STRESS_64
delete CAUCHY_STRESS_02, CAUCHY_STRESS_11, CAUCHY_STRESS_20, CAUCHY_STRESS_29, CAUCHY_STRESS_38, CAUCHY_STRESS_47, >
       CAUCHY_STRESS_56, CAUCHY_STRESS_65
delete CAUCHY_STRESS_03, CAUCHY_STRESS_12, CAUCHY_STRESS_21, CAUCHY_STRESS_30, CAUCHY_STRESS_39, CAUCHY_STRESS_48, >
       CAUCHY_STRESS_57, CAUCHY_STRESS_66
delete CAUCHY_STRESS_04, CAUCHY_STRESS_13, CAUCHY_STRESS_22, CAUCHY_STRESS_31, CAUCHY_STRESS_40, CAUCHY_STRESS_49, >
       CAUCHY_STRESS_58, CAUCHY_STRESS_67
delete CAUCHY_STRESS_05, CAUCHY_STRESS_14, CAUCHY_STRESS_23, CAUCHY_STRESS_32, CAUCHY_STRESS_41, CAUCHY_STRESS_50, >
       CAUCHY_STRESS_59, CAUCHY_STRESS_68
delete CAUCHY_STRESS_06, CAUCHY_STRESS_15, CAUCHY_STRESS_24, CAUCHY_STRESS_33, CAUCHY_STRESS_42, CAUCHY_STRESS_51, >
       CAUCHY_STRESS_60, CAUCHY_STRESS_69
delete CAUCHY_STRESS_07, CAUCHY_STRESS_16, CAUCHY_STRESS_25, CAUCHY_STRESS_34, CAUCHY_STRESS_43, CAUCHY_STRESS_52, >
       CAUCHY_STRESS_61, CAUCHY_STRESS_70
delete CAUCHY_STRESS_08, CAUCHY_STRESS_17, CAUCHY_STRESS_26, CAUCHY_STRESS_35, CAUCHY_STRESS_44, CAUCHY_STRESS_53, >
       CAUCHY_STRESS_62, CAUCHY_STRESS_71
delete CAUCHY_STRESS_09, CAUCHY_STRESS_18, CAUCHY_STRESS_27, CAUCHY_STRESS_36, CAUCHY_STRESS_45, CAUCHY_STRESS_54, >
       CAUCHY_STRESS_63, CAUCHY_STRESS_72

delete FP_01, FP_10, FP_19, FP_28, FP_37, FP_46, FP_55, FP_64
delete FP_02, FP_11, FP_20, FP_29, FP_38, FP_47, FP_56, FP_65
delete FP_03, FP_12, FP_21, FP_30, FP_39, FP_48, FP_57, FP_66
delete FP_04, FP_13, FP_22, FP_31, FP_40, FP_49, FP_58, FP_67
delete FP_05, FP_14, FP_23, FP_32, FP_41, FP_50, FP_59, FP_68
delete FP_06, FP_15, FP_24, FP_33, FP_42, FP_51, FP_60, FP_69
delete FP_07, FP_16, FP_25, FP_34, FP_43, FP_52, FP_61, FP_70
delete FP_08, FP_17, FP_26, FP_35, FP_44, FP_53, FP_62, FP_71
delete FP_09, FP_18, FP_27, FP_36, FP_45, FP_54, FP_63, FP_72


exit
