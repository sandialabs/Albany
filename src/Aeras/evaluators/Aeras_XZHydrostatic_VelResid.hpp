//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_XZHYDROSTATIC_VELRESID_HPP
#define AERAS_XZHYDROSTATIC_VELRESID_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

namespace Aeras {
/** \brief XZHydrostatic equation Residual for atmospheric modeling

    This evaluator computes the residual of the XZHydrostatic equation for
    atmospheric dynamics.

*/

template<typename EvalT, typename Traits>
class XZHydrostatic_VelResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  XZHydrostatic_VelResid(const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Aeras::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint>         wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim,Dim> wGradGradBF;

  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  keGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  PhiGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  etadotdVelx;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  pGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  uDot;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level,Dim>  DVelx;
  PHX::MDField<ScalarT,Cell,QuadPoint,Level>      LaplaceVelx;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim>        density;

  // Output:
  PHX::MDField<ScalarT,Cell,Level,Node> Residual;

  const double viscosity;
  const double hyperviscosity;
  const int numNodes;
  const int numQPs;
  const int numDims;
  const int numLevels;
};
}

#endif
