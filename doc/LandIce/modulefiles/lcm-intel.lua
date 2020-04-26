whatis("LCM Intel compiler")

load("intel/18.0.3.222")

setenv("TOOL_CHAIN", "intel")

setenv("CC", "icc")
setenv("CXX", "icpc")
setenv("FC", "ifort")
