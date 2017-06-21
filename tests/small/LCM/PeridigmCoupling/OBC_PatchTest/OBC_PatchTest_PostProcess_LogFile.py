#!/usr/bin/env python

import string

if __name__ == "__main__":

    inFileName = "log_np5.txt"
    inFile = open(inFileName, 'r')
    lines = inFile.readlines()
    inFile.close()

    obc_vals = []

    trigger = False
    iteration = 0

    for line in lines:

        if "Quasi-Newton Method" in line:
            trigger = True

        if trigger == True:
            vals = string.splitfields(line)
            if len(vals) > 0:
                first_val = string.strip(vals[0])
                if first_val == str(iteration):
                    iteration += 1
                    obc_vals.append(float(vals[1]))

    outFileName = "functional_np5.txt"
    outFile = open(outFileName, 'w')
    print
    for i in range(len(obc_vals)):
        val = obc_vals[i]
        outFile.write(str(i+1) + " " + str(val) + "\n")
        print str(i+1) + " " + str(val)
    outFile.close()

    print "\nData written to " + outFileName + "\n"
