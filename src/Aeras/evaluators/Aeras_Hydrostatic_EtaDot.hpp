//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_HYDROSTATIC_ETADOT_HPP
#define AERAS_HYDROSTATIC_ETADOT_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {

template<typename EvalT, typename Traits>
class Hydrostatic_EtaDot : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  Hydrostatic_EtaDot(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim>    sphere_coord;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Level>      pressure;
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      etadot;

  const int numQPs;
  const int numDims;
  const int numLevels;

  bool pureAdvection;
  
  enum ADVTYPE {UNKNOWN, PRESCRIBED_1_1, PRESCRIBED_1_2};
  ADVTYPE adv_type;

};
}

#endif
