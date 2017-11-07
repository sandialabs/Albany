//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ROTATING_REFERENCE_FRAME_HPP
#define ROTATING_REFERENCE_FRAME_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"

namespace AFRL {

template<typename EvalT, typename Traits>
class RotatingReferenceFrame :
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits> {

public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  RotatingReferenceFrame(Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);
  virtual ~RotatingReferenceFrame();

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:

//! Validate the name strings under "Rotating Reference Frame" section in xml input file,
  Teuchos::RCP<const Teuchos::ParameterList>
               getValidRotatingReferenceFrameParameters() const;

  // Input:
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordinates;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;
  PHX::MDField<ScalarT,Cell> density;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> force;

  std::size_t numQPs;

  double axisOrigin[3];
  double axisDirection[3];
  double angularFrequency;
};
}

#endif
