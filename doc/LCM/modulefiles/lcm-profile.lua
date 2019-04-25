whatis("LCM profile build type")

setenv("BUILD_TYPE", "profile")
setenv("BUILD_STRING", "RELWITHDEBINFO")
setenv("LCM_FPE_SWITCH", "OFF")
setenv("LCM_DENORMAL_SWITCH", "OFF")
if (isloaded("lcm-gcc")) then
   setenv("LCM_FPE_SWITCH", "ON")
   setenv("LCM_DENORMAL_SWITCH", "ON")
end
setenv("LCM_CXX_FLAGS", "-g -O3")
