import numpy as np
import unittest
import subprocess
import os.path
from matrix_reader import read_mm
import sys


class TestHessian(unittest.TestCase):
    AlbanyEXE=''
    MPIEXE=''
    YAMLFILE=''
    def test_hessian(self):
        if os.path.isfile('H-000.mm'):
            os.remove('H-000.mm')

        subprocess.call(self.AlbanyEXE+' '+self.YAMLFILE, shell=True)

        H = read_mm("H-000.mm")
        H_ref = read_mm("H-ref-000.mm")

        diff = H_ref-H
        largest_diff = np.amax(np.abs(diff))

        tol = 1e-8
        self.assertTrue(largest_diff < tol)


if __name__ == '__main__':

    TestHessian.YAMLFILE = sys.argv.pop()
    # The AlbanyEXE argument can look like this ".../mpiexec;-np;1;/...", so we have to replace the potential ';' by ' ':
    TestHessian.AlbanyEXE = sys.argv.pop().replace(';', ' ')

    unittest.main()
