whatis("LCM wrap-up environment definitions")

local arch = os.getenv("ARCH")
local tool_chain = os.getenv("TOOL_CHAIN")
local build_type = os.getenv("BUILD_TYPE")
local build = arch .. tool_chain .. build_type
local lcm_dir = os.getenv("LCM_DIR")
local install_dir = lcm_dir .. "/trilinos-install-" .. build
local alb_exe_dir = lcm_dir .. "/albany-build-" .. build .. "/src"

setenv("BUILD", build)
setenv("INSTALL_DIR", install_dir)
setenv("OMPI_CC", os.getenv("CC"))
setenv("OMPI_CXX", os.getenv("CXX"))
setenv("OMPI_FC", os.getenv("FC"))

prepend_path("LD_LIBRARY_PATH", install_dir .. "/lib")
prepend_path("PATH", alb_exe_dir)
prepend_path("PATH", alb_exe_dir .. "/LCM")
prepend_path("PATH", install_dir .. "/bin")
prepend_path("PYTHONPATH", install_dir .. "/lib")
