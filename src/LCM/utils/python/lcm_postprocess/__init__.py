import sys
import os
import contextlib
import cStringIO
import time
    
# add Cubit libraries to your path
sys.path.append('/home/callema/cubit/bin')
import cubit

#
# Local classes
#
class Timer:  

    def __enter__(self):
        self.start = time.clock()
        self.now = self.start
        self.last = self.now
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

    def check(self):
        self.now = time.clock()
        self.step = self.now - self.last
        self.interval = self.now - self.start
        self.last = self.now

#
# Local functions
#
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