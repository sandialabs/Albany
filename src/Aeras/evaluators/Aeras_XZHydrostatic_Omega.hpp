//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_OMEGA_HPP
#define AERAS_XZHYDROSTATIC_OMEGA_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {
/** \brief Density for XZHydrostatic atmospheric model

    This evaluator computes the density for the XZHydrostatic model
    of atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_Omega : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_Omega(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint>     Velx;
  PHX::MDField<ScalarT,Cell,QuadPoint>     density;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> gradp;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> gradpivelx;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint>      omega;

  const int numQPs;
  const int numLevels;
  double Cp ;

};
}

#endif
