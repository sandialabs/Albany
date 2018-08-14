//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "ParallelConstitutiveModel_Def.hpp"
#include "ParallelNeohookeanModel.hpp"
#include "ParallelNeohookeanModel_Def.hpp"

template <typename EvalT, typename Traits>
LCM::ParallelNeohookeanModel<EvalT, Traits>::ParallelNeohookeanModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ParallelConstitutiveModel<
          EvalT,
          Traits,
          NeohookeanKernel<EvalT, Traits>>(p, dl)
{
}

PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::NeohookeanKernel)
PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::ParallelNeohookeanModel)
