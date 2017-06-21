#! /usr/bin/env python

import sys

from runtest import runtest

result = 0

print "test 1 - uniaxial"
name = "Neohookean-uniaxial"
result = runtest(name)
if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

print "test 2 - shear"
name = "Neohookean-shear"
result = runtest(name)
if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

print "test 3 - hydrostatic"
name = "Neohookean-hydrostatic"
result = runtest(name)
if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

print "test 3 - general"
name = "Neohookean-general"
result = runtest(name)
if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

sys.exit(result)
