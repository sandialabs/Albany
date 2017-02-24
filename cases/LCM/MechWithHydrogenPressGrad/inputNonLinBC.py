Count = 0.0
outFileName = 'inputNonLinBC.xml'
dataFile = open(outFileName, 'w')

C0 = 0.0
C1 = 4.0e-3
C2 = 1.6e-2
C3 = 1.0e-2
Count = 0.0

for nodeset in range(7,48):
     dispVal = C0 + C1*Count + C2*Count**2 + C3*Count**3
     dataFile.write('      <ParameterList name="Time Dependent DBC on NS nodelist_' + str(nodeset) +  ' for DOF X">' + "\n")
     dataFile.write('        <Parameter name="Number of points" type="int" value="3"/>' + "\n")
     dataFile.write('        <Parameter name="Time Values" type="Array(double)" value="{ 0.0, 1000.0, 2000.0}"/>' + "\n")
     dataFile.write('        <Parameter name="BC Values"   type="Array(double)" value="{ 0.0, ' + str(dispVal) + ', ' + str(dispVal) + '}"/>' + "\n")
     dataFile.write('      </ParameterList> ' + "\n")
     Count += 0.025
dataFile.close()
print "data written to", outFileName

print
    
