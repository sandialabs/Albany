whatis("LCM Intel compiler")

load("intel/17.0.4.196")

setenv("TOOL_CHAIN", "intel")

setenv("CC", "icc")
setenv("CXX", "icpc")
setenv("FC", "ifort")
