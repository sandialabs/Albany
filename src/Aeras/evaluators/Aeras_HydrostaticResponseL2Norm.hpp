//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_HYDROSTATIC_RESPONSE_L2NORM_HPP
#define AERAS_HYDROSTATIC_RESPONSE_L2NORM_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras{

template<typename EvalT, typename Traits>
class HydrostaticResponseL2Norm :
  public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
{
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  HydrostaticResponseL2Norm(Teuchos::ParameterList& p,
                       const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                                   PHX::FieldManager<Traits>& vm);

  void preEvaluate(typename Traits::PreEvalData d);

  void evaluateFields(typename Traits::EvalData d);

  void postEvaluate(typename Traits::PostEvalData d);

private:
  Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

  std::size_t numQPs;
  std::size_t numDims;

  PHX::MDField<ScalarT> field;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim> velocity;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level> temperature;
  const int numLevels;
};

}

#endif




