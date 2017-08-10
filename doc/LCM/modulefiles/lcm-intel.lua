whatis("LCM Intel compiler")

load("intel/17.0.4.196")

setenv("TOOL_CHAIN", "intel")

setenv("CC", subprocess("which icc"))
setenv("CXX", subprocess("which icpc"))
setenv("FC", subprocess("which ifort"))
