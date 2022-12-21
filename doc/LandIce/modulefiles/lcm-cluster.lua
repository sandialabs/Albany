whatis("LCM cluster (TOSS3) base environment")

setenv("LCM_ENV_TYPE", "cluster-toss3")

unload("openmpi-intel/1.10")
unload("intel/16.0")

-- /usr/bin/python is 2.7.5, good enough
-- /usr/bin/git is 2.7.4, good enough
load("cmake/3.11.1")
-- load("cde/dev/compiler/gcc/7.2.0")
-- load("cde/dev/cmake/3.19.2")
-- load("cde/prod/compiler/gcc/7.2.0")
load("cde/x86_64/v2/compiler/gcc/7.2.0")
load("cmake/3.22.3")

conflict("lcm-fedora", "lcm-sems")
