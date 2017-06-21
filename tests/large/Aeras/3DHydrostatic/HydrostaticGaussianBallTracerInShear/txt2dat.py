#!/usr/bin/env python

import sys
import math

# use exo2txt to make an ascii version first...
f = open('sphere10_quad9.txt', 'r')
lines = f.readlines()
i=0
while (-1==lines[i].find("Coordinates")): i = i + 1
i = i + 1
n = 0
x_coord=[]
y_coord=[]
z_coord=[]
while (-1==lines[i].find("Node")):
  words =  lines[i].split()
  x_coord.append(float(words[0]))
  y_coord.append(float(words[1]))
  z_coord.append(float(words[2]))
  n = n + 1
  i = i + 1
numnode = len(x_coord)

while (-1==lines[i].find("Connectivity")): i = i + 1
i = i + 1
exo_connect=[]
while (-1==lines[i].find("Attributes")) :
  words =  lines[i].split()
  for k in range(len(words)) : exo_connect.append(int(words[k])-1)
  i = i + 1

connect=[]
map=[]
map.append([0,4,8,7])
map.append([4,1,5,8])
map.append([7,8,6,3])
map.append([8,5,2,6])
for k in range(0, len(exo_connect),9) : 
  for n in range(0,4) : 
    for m in range(0,4) : 
      connect.append(1+exo_connect[k+map[n][m]])
numcon = len(connect)/4

while (-1==lines[i].find("Variable names")): i = i + 1
i = i + 1
words = lines[i].split()
numvar = int(words[1])
numlev = (numvar-1)/5
numtr  = (numvar-1)/numlev - 3

for lev in range(1,numlev) :
  for i in range(numnode) :
    r     = math.sqrt(x_coord[i]*x_coord[i] + y_coord[i]*y_coord[i] + z_coord[i]*z_coord[i]) 
    theta = math.acos(z_coord[i]/r) 
    phi   = math.atan2(y_coord[i],x_coord[i])
    r     = r+lev
    x     = r*math.sin(theta)*math.cos(phi)
    y     = r*math.sin(theta)*math.sin(phi)
    z     = r*math.cos(theta)
    x_coord.append(x)
    y_coord.append(y)
    z_coord.append(z)
  
print "TITLE=\"State variables at t = 0.0000e+00, step = 0\""
names = "VARIABLES=\"x\",\"y\",\"z\""
names += ",\"p\",\"Ux\",\"Uy\",\"T\""
for l in range(numtr) :
  names += ",\"Tr_" + str(l) + "\""
print names

print >> sys.stderr, "txttodat:",
while (-1==lines[i].find("Nodal variables")): i = i + 1
i = i + 1
time = 0
while i<len(lines) :
  var = []
  while (i<len(lines) and -1==lines[i].find("Time step")):
    n = 0
    j = len(var)
    var.append([])
    while (n < numvar) :
      words = lines[i].split()
      i = i + 1
      for k in range(len(words)): var[j].append(float(words[k]))
      n = n + len(words)
  for k in range(numlev) :
    print "ZONE T=\"Element ",k+1,"\", N="+str(numnode)+", E="+str(numcon)+", DATAPACKING=BLOCK, ZONETYPE=FEQUADRILATERAL, SOLUTIONTIME="+str(time)

    for n in range(numnode) :
      print repr(x_coord[k*numnode+n]).rjust(11),
      if (0==(n+1)%10) : print
    if (0!=numnode%10) : print

    for n in range(numnode) :
      print repr(y_coord[k*numnode+n]).rjust(11),
      if (0==(n+1)%10) : print
    if (0!=numnode%10) : print

    for n in range(numnode) :
      print repr(z_coord[k*numnode+n]).rjust(11),
      if (0==(n+1)%10) : print
    if (0!=numnode%10) : print

    for n in range(numnode) :
      print var[n][0],
      if (0==(n+1)%10) : print
    if (0!=numnode%10) : print

    for t in range(1,3) :
      for n in range(numnode) :
        print var[n][k*3+t],
        if (0==(n+1)%10) : print
      if (0!=numnode%10) : print

    for n in range(numnode) :
      print var[n][(k+1)*3],
      if (0==(n+1)%10) : print
    if (0!=numnode%10) : print

    for t in range(numtr) :
      for n in range(numnode) :
        print var[n][3*numlev+1+t],
        if (0==(n+1)%10) : print
      if (0!=numnode%10) : print

    for n in range(0,4*numcon,4) :
      print connect[n+0],connect[n+1],connect[n+2],connect[n+3]

  print >> sys.stderr, time, 
  time = time+1
  while (i < len(lines) and -1==lines[i].find("Nodal variables")): i = i + 1
  i = i + 1


print >> sys.stderr
