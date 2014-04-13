//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DAKOTA_H
#define ALBANY_DAKOTA_H

#ifdef ALBANY_DAKOTA

/** \brief Main routine to drive ModelEvaluator application with Dakota */
int Albany_Dakota(int argc, char *argv[]);

#else // ALBANY_DAKOTA
int Albany_Dakota(int argc, char *argv[])
{
  std::cout << "\nDakota requested but not compiled in!\n" << std::endl;
  return 999;
}
#endif  // ALBANY_DAKOTA
#endif //ALBANY_DAKOTA_H
