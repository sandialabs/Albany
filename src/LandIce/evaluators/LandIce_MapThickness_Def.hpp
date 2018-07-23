//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "PHAL_Utilities.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits>
MapThickness<EvalT, Traits>::
MapThickness (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :

  H_in(p.get<std::string> ("Input Thickness Name"), dl->node_scalar),
  H_obs(p.get<std::string> ("Observed Thickness Name"), dl->node_scalar),
  bed(p.get<std::string> ("Observed Bed Topography Name"), dl->node_scalar),
  H_out(p.get<std::string> ("Output Thickness Name"), dl->node_scalar),
  H_min(p.get<std::string> ("Thickness Lower Bound Name"), dl->node_scalar),
  H_max(p.get<std::string> ("Thickness Upper Bound Name"), dl->node_scalar)
{
  this->addDependentField(H_in);
  this->addDependentField(bed);
  this->addDependentField(H_obs);
  this->addDependentField(H_min);
  this->addDependentField(H_max);
  this->addEvaluatedField(H_out);

  this->setName("MapThickness"+PHX::typeAsString<EvalT>());
}

template<typename EvalT, typename Traits>
void MapThickness<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(H_in,fm);
  this->utils.setFieldData(H_min,fm);
  this->utils.setFieldData(H_max,fm);
  this->utils.setFieldData(H_out,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MapThickness<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  PHAL::MDFieldIterator<const MeshScalarT> Hin(H_in);
  PHAL::MDFieldIterator<const MeshScalarT> Hmin(H_min);
  PHAL::MDFieldIterator<const MeshScalarT> Hmax(H_max);
  PHAL::MDFieldIterator<const MeshScalarT> Hobs(H_obs);
  PHAL::MDFieldIterator<const MeshScalarT> b(bed);
  PHAL::MDFieldIterator<MeshScalarT> Hout(H_out);
  for (; !Hin.done(); ++Hin, ++Hmin, ++Hmax, ++Hout, ++b, ++Hobs) {
    MeshScalarT tmp = std::tanh(*Hin);
    *Hout = (*Hmax+*Hmin)/2.0+(*Hmax-*Hmin)/2.0*tmp;
  }
}

} // Namespace LandIce
