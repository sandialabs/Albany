//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_TOTALVOLUME_HPP
#define AERAS_TOTALVOLUME_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras{

template<typename EvalT, typename Traits>
class TotalVolume :
  public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
{
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  TotalVolume(Teuchos::ParameterList& p,
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

  PHX::MDField<const ScalarT> field;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level> density;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level,Dim> velocity;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level> temperature;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level> Cpstar;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level> pie;
  const int numLevels;
  double Phi0;
};

}

#endif




