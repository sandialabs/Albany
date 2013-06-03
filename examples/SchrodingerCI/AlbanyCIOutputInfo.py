import sys

def parseAlbanyCI_stdoe(filename, bDebug=False):
    bInCIBlock = False
    bInEvalBlock = False
    blockInfo = [ ]

    for line in open(filename):
        line = line.strip()
        if bInCIBlock:
            if line.find("Matrix Block:") >= 0:
                curMatrixBlock = line
                i = line.find("####   Matrix Block:") + len("####   Matrix Block:")
                end = line[i:].find("####")
                part = line[i:i+end]
                blockVals = { 'evals': [ ] }
                for keyEqualsVal in part.split():
                    keyVal = keyEqualsVal.split('=')
                    if len(keyVal) == 2: 
                        blockVals[keyVal[0]] = int(keyVal[1])  #Note: so far all values are *integers*
                blockInfo.append(blockVals)
                if bDebug: print "Matrix Block Line: ",line
            elif len(curMatrixBlock) > 0:
                if line.find("Eigenvalue") >= 0 and line.find("Direct Residual") >= 0:
                    bInEvalBlock = True
                    nEvals = 0

                #This elif block is for backward compatibility (to remove later) when dim=1 matrix eigenvalues
                #  were printed differently than for dim > 1 matrices by AlbanyCI.
                elif line.find("Eigenvalue") >= 0:
                    evalLine = line.split()
                    if bDebug: print "Single Eval = ", evalLine[1]
                    blockVals['evals'].append(float(evalLine[1]))
                    nEvals = 0

                elif bInEvalBlock:
                    if line.find("----------") >= 0:
                        if nEvals > 0: bInEvalBlock = False
                    else:
                        if bDebug: print "Eval %d = " % nEvals,line
                        evalAndResid = line.split()
                        blockVals['evals'].append(float(evalAndResid[0]))
                        nEvals += 1
                elif line.find("CI solve finished") >= 0:
                    bInCIBlock = False

        elif line.find("AlbanyCI Solver called") >= 0:
            bInCIBlock = True
            curMatrixBlock = ""

    if bDebug: print "Collected info:\n", blockInfo
    return blockInfo


def getExchangeEnergy(blockInfo):
    singlet_energy = None
    triplet_energy = None
    for b in blockInfo:
        if b.get('2Sz',-1) == 0:
            if b.has_key('2S'):
                if b['2S'] == 0: singlet_energy = b['evals'][0]
                elif b['2S'] == 2: triplet_energy = b['evals'][0]
            else:
                #When S2 value is not known, assume S and T are two lowest Sz=0 states
                singlet_energy = b['evals'][0]
                triplet_energy = b['evals'][1]

    if singlet_energy is None or triplet_energy is None:
        raise ValueError("Could not find either or both of singlet and triplet energies")
    
    return triplet_energy - singlet_energy
                

def test(argv):

    if len(argv) < 1+1:
        print "This is a unit test for this module, which reads the eigenvalues from an AlbanyCI (stdout) output file."
        print "Usage: <AlbanyCI Output Filename>"
        exit(-1)

    blockInfo = parseAlbanyCI_stdoe(argv[1])
    for b in blockInfo:
        print "Eigenvalues for ",
        for k in b:
            if k == "evals": continue
            print "%s=%s " % (k,b[k]),
        print "block:"
        for ev in b['evals']: 
            print ev
        print ""


if __name__ == "__main__":
    test(sys.argv)
    exit(0)

