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
def stdout_redirected(to = os.devnull):
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
            _redirect_stdout(to = file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to = old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different