#!/usr/bin/env python

import sys

# use exo2txt to make an ascii version first...
f = open('xzhydrostatic.txt', 'r')
lines = f.readlines()
i=0
while (-1==lines[i].find("Coordinates")): i = i + 1
i = i + 1
n = 0
x_coord=[]
while (-1==lines[i].find("Node")):
  x_coord.append(float(lines[i]))
  n = n + 1
  i = i + 1
numelm = len(x_coord)
while (-1==lines[i].find("Variable names")): i = i + 1
i = i + 1
words = lines[i].split()
numvar = int(words[1])
numlev = (numvar-1)/5
numtr  = (numvar-1)/numlev - 2

print "TITLE=\"State variables at t = 0.0000e+00, step = 0\""
names = "VARIABLES=\"x\",\"y\",\"p\""
names += ",\"U\",\"T\""
for l in range(numtr) :
  names += ",\"Tr_" + str(l) + "\""
print names

print >> sys.stderr, "txttodat:",
while (-1==lines[i].find("Nodal variables")): i = i + 1
i = i + 1
time = 0
element = 0
while i<len(lines) :
  var = []
  while (i<len(lines) and -1==lines[i].find("Time step")):
    n = 0
    j = len(var)
    var.append([])
    while (n < numvar) :
      words = lines[i].split()
      i = i + 1
      for k in range(len(words)): 
        if words[k] == "NaN" : sys.exit ("\nERROR: Found nan.")
        if 0<words[k].find('+') and words[k].find('E+')<0: sys.exit ("\nERROR: Found invalid float.")
      for k in range(len(words)): var[j].append(float(words[k]))
      n = n + len(words)
  element = element + 1

  print "ZONE T=\"Element ",element,"\", I="+str(numelm)+", J="+str(numlev)+", F=BLOCK, SOLUTIONTIME="+str(time)
  for k in range(numlev) :
    for n in range(numelm) :
      print x_coord[n],
      if (0==(n+1)%10) : print
    if (0!=numelm%10) : print

  for k in range(numlev) :
    for n in range(numelm) :
      print numlev-k-1,
      if (0==(n+1)%10) : print
    if (0!=numelm%10) : print

  for k in range(numlev) :
    for n in range(numelm) :
      print var[n][0],
      if (0==(n+1)%10) : print
    if (0!=numelm%10) : print

  for k in range(numlev) :
    for n in range(numelm) :
      print var[n][2*k+1],
      if (0==(n+1)%10) : print
    if (0!=numelm%10) : print

  for k in range(numlev) :
    for n in range(numelm) :
      print var[n][2*k+2],
      if (0==(n+1)%10) : print
    if (0!=numelm%10) : print

  for t in range(1+2*numlev,numvar,numlev) :
    for k in range(numlev) :
      for n in range(numelm) :
        print var[n][k+t],
        if (0==(n+1)%10) : print
      if (0!=numelm%10) : print

  print >> sys.stderr, time, 
  time = time+1
  while (i < len(lines) and -1==lines[i].find("Nodal variables")): i = i + 1
  i = i + 1


print >> sys.stderr
