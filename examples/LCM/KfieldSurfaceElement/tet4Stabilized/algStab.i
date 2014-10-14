save all

localPressure = (nodal_Cauchy_Stress_1 + nodal_Cauchy_Stress_5 + nodal_Cauchy_Stress_9)/3.

stress_11 = nodal_Cauchy_Stress_1
stress_12 = nodal_Cauchy_Stress_2
stress_13 = nodal_Cauchy_Stress_3
stress_21 = nodal_Cauchy_Stress_4
stress_22 = nodal_Cauchy_Stress_5
stress_23 = nodal_Cauchy_Stress_6
stress_31 = nodal_Cauchy_Stress_7
stress_32 = nodal_Cauchy_Stress_8
stress_33 = nodal_Cauchy_Stress_9

stab_stress_11 = nodal_Cauchy_Stress_1 - localPressure + pressure
stab_stress_12 = nodal_Cauchy_Stress_2
stab_stress_13 = nodal_Cauchy_Stress_3
stab_stress_21 = nodal_Cauchy_Stress_4
stab_stress_22 = nodal_Cauchy_Stress_5 - localPressure + pressure
stab_stress_23 = nodal_Cauchy_Stress_6
stab_stress_31 = nodal_Cauchy_Stress_7
stab_stress_32 = nodal_Cauchy_Stress_8
stab_stress_33 = nodal_Cauchy_Stress_9 - localPressure + pressure

exit
