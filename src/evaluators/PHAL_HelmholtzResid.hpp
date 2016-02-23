//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_HELMHOLTZRESID_HPP
#define PHAL_HELMHOLTZRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Sacado_ParameterRegistration.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class HelmholtzResid : public PHX::EvaluatorWithBaseImpl<Traits>,
 		       public PHX::EvaluatorDerived<EvalT, Traits>,
                       public Sacado::ParameterAccessor<EvalT, SPL_Traits>  {


public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  HelmholtzResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  virtual ScalarT& getValue(const std::string &n) {return ksqr;};

private:

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<ScalarT,Cell,QuadPoint> U;
  PHX::MDField<ScalarT,Cell,QuadPoint> V;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> UGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> VGrad;

  PHX::MDField<ScalarT,Cell,QuadPoint> USource;
  PHX::MDField<ScalarT,Cell,QuadPoint> VSource;

  bool haveSource;

  ScalarT ksqr;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> UResidual;
  PHX::MDField<ScalarT,Cell,Node> VResidual;
};
}

#endif
