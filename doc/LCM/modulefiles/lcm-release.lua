whatis("LCM release build type")

setenv("BUILD_TYPE", "release")
setenv("BUILD_STRING", "RELEASE")
setenv("LCM_FPE_SWITCH", "ON")
setenv("LCM_DENORMAL_SWITCH", "ON")
setenv("LCM_CXX_FLAGS", "-DNDEBUG")
