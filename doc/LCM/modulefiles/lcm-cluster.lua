whatis("LCM cluster (TOSS3) base environment")

setenv("LCM_ENV_TYPE", "cluster-toss3")

unload("openmpi-intel/1.10")
unload("intel/16.0")

-- /usr/bin/python is 2.7.5, good enough
-- /usr/bin/git is 2.7.4, good enough
load("cmake/3.8")

conflict("lcm-fedora", "lcm-sems")
