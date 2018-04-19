//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "J2MiniSolver.hpp"
#include "J2MiniSolver_Def.hpp"
#include "ParallelConstitutiveModel_Def.hpp"

template <typename EvalT, typename Traits>
LCM::J2MiniSolver<EvalT, Traits>::J2MiniSolver(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::
          ParallelConstitutiveModel<EvalT, Traits, J2MiniKernel<EvalT, Traits>>(
              p,
              dl)
{
}

PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::J2MiniKernel)
PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::J2MiniSolver)
