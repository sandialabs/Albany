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
#include "Aeras_Dimension.hpp"

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
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  divpivelx;
  PHX::MDField<ScalarT,Cell,QuadPoint>            pdotP0;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      Pi;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      Temperature;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  Velx;

  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      etadotdT;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  etadotdVelx;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      Pidot;

  std::map<std::string, PHX::MDField<ScalarT,Cell,QuadPoint,Level> > Tracer;
  std::map<std::string, PHX::MDField<ScalarT,Cell,QuadPoint,Level> > etadotdTracer;

  const Teuchos::ArrayRCP<std::string> tracerNames;
  const Teuchos::ArrayRCP<std::string> etadotdtracerNames;

  const int numQPs;
  const int numDims;
  const int numLevels;

};
}

#endif
