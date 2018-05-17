import numpy as np

cauchy_stress = []
for i in range(1,45):
    cauchy_stress.append(inputs[0].CellData['Cauchy_Stress_%02d' % (i,)])
dataArray = np.squeeze(np.asarray(np.matrix(cauchy_stress).max(axis=0)))
output.CellData.append(dataArray, 'Cauchy_Stress_Max')

displacement = inputs[0].PointData["solution_"]
output.PointData.append(displacement, "disp_")
