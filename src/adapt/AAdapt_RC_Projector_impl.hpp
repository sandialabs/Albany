//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_RC_PROJECTOR_IMPL
#define AADAPT_RC_PROJECTOR_IMPL

namespace AAdapt {
namespace rc {

/*! \brief Implement details related to projection for rc::Manager.
 *
 *  For efficient rebuilding, separate out the solver-related code. Including
 *  Ifpack2 and Belos slows build time of a file, and I want
 *  AAdapt_RC_Manager.cpp to build quickly.
 */

//! Solve A x = b using preconditioner P. Construct P if it is null on input.
Teuchos::RCP<Tpetra_MultiVector>
solve(const Teuchos::RCP<const Tpetra_CrsMatrix>& A,
      Teuchos::RCP<Tpetra_Operator>& P,
      const Teuchos::RCP<const Tpetra_MultiVector>& b,
      Teuchos::ParameterList& belos_pl);

} // namespace rc
} // namespace AAdapt

#endif // AADAPT_RC_PROJECTOR_IMPL
