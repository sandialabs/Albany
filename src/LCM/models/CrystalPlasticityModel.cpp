//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_AlbanyTraits.hpp"

#include "CrystalPlasticityModel.hpp"
#include "CrystalPlasticityModel_Def.hpp"
#include "ParallelConstitutiveModel_Def.hpp"

template<typename EvalT, typename Traits>
LCM::CrystalPlasticityModel<EvalT,Traits>::CrystalPlasticityModel(
    Teuchos::ParameterList* p,
    Teuchos::RCP<Albany::Layouts> const & dl)
  : LCM::ParallelConstitutiveModel<
      EvalT, Traits, CrystalPlasticityKernel<EvalT, Traits>>(p, dl)
{}

PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::CrystalPlasticityKernel)
PHAL_INSTANTIATE_TEMPLATE_CLASS(LCM::CrystalPlasticityModel)

