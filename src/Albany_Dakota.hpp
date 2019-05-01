//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DAKOTA_HPP
#define ALBANY_DAKOTA_HPP

#include "Albany_config.h"

#ifdef ALBANY_DAKOTA

/** \brief Main routine to drive ModelEvaluator application with Dakota */
int Albany_Dakota(int argc, char *argv[]);

#else // ALBANY_DAKOTA
#include <stdio.h>
int Albany_Dakota(int /* argc */, char * /*argv*/[])
{
  printf("\nDakota requested but not compiled in!\n");
  return 999;
}

#endif  // ALBANY_DAKOTA

#endif //ALBANY_DAKOTA_HPP
