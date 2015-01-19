//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_RC_PROJECTOR_IMPL
#define AADAPT_RC_PROJECTOR_IMPL

#include "AAdapt_RC_DataTypes.hpp"

// Forward declarations.
namespace PHAL { class Workset; }

namespace AAdapt {
namespace rc {

/*! \brief Implement details related to projection for rc::Manager.
 *
 *  For efficient rebuilding, separate out the solver-related code. Including
 *  Ifpack2 and MueLu tends to slow rebuild time of a file, and I want
 *  AAdapt_RC_Manager.cpp to build quickly.
 */

} // namespace rc
} // namespace AAdapt

#endif // AADAPT_RC_PROJECTOR_IMPL
