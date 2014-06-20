//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_ETADOTPI_HPP
#define AERAS_XZHYDROSTATIC_ETADOTPI_HPP

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
class XZHydrostatic_EtaDotPi : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_EtaDotPi(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim>  gradpivelx;
  PHX::MDField<ScalarT,Cell,QuadPoint>      pdotP0;
  PHX::MDField<ScalarT,Cell,QuadPoint>      Pi;
  PHX::MDField<ScalarT,Cell,QuadPoint>      Temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint>      Velx;

  PHX::MDField<ScalarT,Cell,QuadPoint>  etadotdT;
  PHX::MDField<ScalarT,Cell,QuadPoint>  etadotdVelx;

  std::map<std::string, PHX::MDField<ScalarT,Cell,QuadPoint> > Tracer;
  std::map<std::string, PHX::MDField<ScalarT,Cell,QuadPoint> > etadotdTracer;

  const Teuchos::ArrayRCP<std::string> tracerNames;
  const Teuchos::ArrayRCP<std::string> etadotdtracerNames;

  const int numQPs;
  const int numLevels;

  ScalarT P0;
  ScalarT Ptop;

};
}

#endif
