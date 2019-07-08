//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DEBUG_HPP
#define ALBANY_DEBUG_HPP

// Get Albany configuration macros
#include "Albany_config.h"

// Checks if the previous Kokkos::Cuda kernel has failed
#ifdef ALBANY_CUDA_ERROR_CHECK
#include <stdexcept> // For cudaCheckError
#define cudaCheckError() \
  { cudaError(__FILE__, __LINE__); }
inline void
cudaError(const char* file, int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(
        stderr, "CUDA Error: %s before %s:%d\n", cudaGetErrorString(err), file,
        line);
    throw std::runtime_error(cudaGetErrorString(err));
  }
}
#else
#define cudaCheckError()
#endif

// NVTX Range creates a colored range which can be viewed on the nvvp timeline
// (from Parallel Forall blog)
#ifdef ALBANY_CUDA_NVTX
#include "nvToolsExt.h"
static const uint32_t nvtx_colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00,
    0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static const int num_nvtx_colors = sizeof(nvtx_colors)/sizeof(uint32_t);
#define PUSH_RANGE(name,cid) { \
  int color_id = cid; \
  color_id = color_id%num_nvtx_colors;\
  nvtxEventAttributes_t eventAttrib = {0}; \
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = nvtx_colors[color_id]; \
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
  eventAttrib.message.ascii = name; \
  nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

#ifdef __CUDA_ARCH__
#define ALBANY_ASSERT_IMPL(cond, ...) assert(cond)
#else
#include <iostream>  // For std::cerr
#define ALBANY_ASSERT_IMPL(cond, msg, ...)                \
  do {                                                    \
    if (!(cond)) {                                        \
      std::ostringstream omsg;                            \
      omsg << #cond " failed at ";                        \
      omsg << __FILE__ << " +" << __LINE__ << '\n';       \
      omsg << msg << '\n';                                \
      std::cerr << #cond " failed at "                    \
                << __FILE__ << " +" << __LINE__ << "\n"   \
                << msg << '\n';                           \
      abort ();                                           \
    }                                                     \
  } while (0)
#endif

#define ALBANY_ASSERT(...) ALBANY_ASSERT_IMPL(__VA_ARGS__, "")

#ifdef NDEBUG
#define ALBANY_EXPECT(...)
#else
#define ALBANY_EXPECT(...) ALBANY_ASSERT(__VA_ARGS__)
#endif

#define ALBANY_ALWAYS_ASSERT(cond) ALBANY_ASSERT(cond)
#define ALBANY_ALWAYS_ASSERT_VERBOSE(cond, msg) ALBANY_ASSERT(cond, msg)
#define ALBANY_DEBUG_ASSERT(cond) ALBANY_EXPECT(cond)
#define ALBANY_DEBUG_ASSERT_VERBOSE(cond, msg) ALBANY_EXPECT(cond, msg)

#endif // ALBANY_DEBUG_HPP
