//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PYUTILS_H
#define ALBANY_PYUTILS_H

// Get Albany configuration macros
#include "Albany_config.h"

#include <sstream>

#include "Albany_CommUtils.hpp"
#include "Albany_Macros.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_TpetraTypes.hpp"
#include "Teuchos_RCP.hpp"

namespace PyAlbany
{

  //! Print ascii art and version information for PyAlbany
  void
  PrintPyHeader(std::ostream &os);
} // namespace PyAlbany

#endif // ALBANY_PYUTILS_H
