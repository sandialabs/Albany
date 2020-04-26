whatis("LCM release build type")

setenv("BUILD_TYPE", "release")
setenv("BUILD_STRING", "RELEASE")
setenv("LCM_FPE_SWITCH", "OFF")
setenv("LCM_DENORMAL_SWITCH", "OFF")
if (isloaded("lcm-gcc")) then
   setenv("LCM_FPE_SWITCH", "ON")
   setenv("LCM_DENORMAL_SWITCH", "ON")
end
setenv("LCM_CXX_FLAGS", "-DNDEBUG")
