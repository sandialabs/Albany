import sys
import os
import contextlib
import cStringIO
    
# add Cubit libraries to your path
sys.path.append('/home/callema/cubit/bin')
import cubit

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    yield
    sys.stdout = save_stdout

@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

def createMesh(nameFileBase, nElementsCoarse, nElementsFine, sizeOverlap):

    with stdout_redirected():
    
        #start cubit - this step is key
        cubit.init([''])
        
        sizeElementFine = 1.0 / float(nElementsFine)
        sizeElementCoarse = 1.0 / float(nElementsCoarse)
    
        cubit.cmd('reset')
        cubit.cmd('undo on')
        cubit.cmd('create brick x 1.0 y 1.0 z ' + str(1.0 + sizeOverlap))
        cubit.cmd('move volume 1 z '+str(1.0 - sizeOverlap) +' include_merged')
        cubit.cmd('volume 1 size '+str(sizeElementCoarse))
        cubit.cmd('mesh volume 1')
        cubit.cmd('block 1 volume 1')
        cubit.cmd('block 1 name "coarse"')
        cubit.cmd('nodeset 1 surface 6')
        cubit.cmd('nodeset 2 surface 4')
        cubit.cmd('nodeset 3 surface 3')
        cubit.cmd('nodeset 4 surface 5')
        cubit.cmd('nodeset 5 surface 2')
        cubit.cmd('nodeset 6 surface 1')
        cubit.cmd('set large exodus file off')
        cubit.cmd('export mesh "' + str(nameFileBase) + '_Cube1.g" overwrite')
        
        cubit.cmd('reset')
        cubit.cmd('undo on')
        cubit.cmd('create brick x 1.0 y 1.0 z ' + str(1.0 + sizeOverlap))
        cubit.cmd('volume 1 size '+str(sizeElementFine))
        cubit.cmd('mesh volume 1')
        cubit.cmd('block 1 volume 1')
        cubit.cmd('block 1 name "fine"')
        cubit.cmd('nodeset 1 surface 6')
        cubit.cmd('nodeset 2 surface 4')
        cubit.cmd('nodeset 3 surface 3')
        cubit.cmd('nodeset 4 surface 5')
        cubit.cmd('nodeset 5 surface 2')
        cubit.cmd('nodeset 6 surface 1')
        cubit.cmd('set large exodus file off')
        cubit.cmd('export mesh "' + str(nameFileBase) + '_Cube0.g" overwrite')
        
    files = os.listdir(os.getcwd())
    
    filesCubit = [file for file in files if file.find('.jou') != -1]
    
    for file in filesCubit:
        os.remove(file)
    
#    outfile = open(nameFileBase+'_createMesh.out', 'w')
#    outfile.write(output)
#    outfile.close()
    
    
    

if __name__ == '__main__':

    if (len(sys.argv) == 1):
        print 'Usage: python createMesh_Cubes.py [base file name, number of coarse element divisions, number of fine element divisions, overlap size]'
        sys.exit()
    elif (len(sys.argv) == 5):
        nameFileBase = sys.argv[1]
        nElementsCoarse = int(sys.argv[2])
        nElementsFine = int(sys.argv[3])
        sizeOverlap = float(sys.argv[4])
    else:
        print 'Incorrect usage'
        print 'Usage: python createMesh_Cubes.py [base file name, number of coarse element divisions, number of fine element divisions, overlap size]'
        sys.exit()
    
    createMesh(nameFileBase, nElementsCoarse, nElementsFine, sizeOverlap)

    
