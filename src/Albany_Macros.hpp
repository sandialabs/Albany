//
// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
//

#ifndef ALBANY_DEBUG_HPP
#define ALBANY_DEBUG_HPP

// Get Albany configuration macros
#include "Albany_config.h"

// NVTX Range creates a colored range which can be viewed on the nvvp timeline
// (from Parallel Forall blog)
#define PUSH_RANGE(name, cid)
#define POP_RANGE

#include <iostream>  // For std::cerr
#define ALBANY_ASSERT(cond, msg)                          \
  do {                                                        \
    if (!(cond)) {                                            \
      std::cerr << #cond " ALBANY_ASSERT failed at ";         \
      std::cerr << __FILE__ << " +" << __LINE__ << "\n";      \
      std::cerr << msg << '\n';                               \
      abort();                                                \
    }                                                         \
  } while (0)

#define ALBANY_ABORT(msg)                                     \
  do {                                                        \
    std::cerr << " ALBANY_ABORT statement at ";               \
    std::cerr << __FILE__ << " +" << __LINE__ << "\n";        \
    std::cerr << msg << '\n';                                 \
    abort();                                                  \
  } while (0)

#define ALBANY_TRACE(msg)                                     \
  do {                                                        \
    std::cout << "********** ALBANY_TRACE at ";               \
    std::cout << __FILE__ << " +" << __LINE__ << "\n";        \
    std::cout << msg << '\n';                                 \
  } while (0)

#if defined(NDEBUG)
#define ALBANY_EXPECT(cond,msg)
#else
#define ALBANY_EXPECT(cond,msg) ALBANY_ASSERT(cond,msg)
#endif

#define ALBANY_DEBUG_ASSERT(cond,msg)       ALBANY_EXPECT(cond, msg)
#define ALBANY_DEBUG_ASSERT_MSG(cond, msg)  ALBANY_EXPECT(cond, msg)

#endif  // ALBANY_DEBUG_HPP
