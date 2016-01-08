//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_NSNEUTRONEQRESID_HPP
#define PHAL_NSNEUTRONEQRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
namespace PHAL {

template<typename EvalT, typename Traits>
class NSNeutronEqResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  NSNeutronEqResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<ScalarT,Cell,QuadPoint> Neutron;
  PHX::MDField<ScalarT,Cell,QuadPoint> NeutronDiff;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> NGrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> Source;
  PHX::MDField<ScalarT,Cell,QuadPoint> Absorp;
  PHX::MDField<ScalarT,Cell,QuadPoint> Fission;  
  PHX::MDField<ScalarT,Cell,QuadPoint> nu;  

  // Output:
  PHX::MDField<ScalarT,Cell,Node> NResidual;

  bool haveNeutSource;
  unsigned int numQPs, numDims, numNodes;
  Intrepid2::FieldContainer<ScalarT> flux;
  Intrepid2::FieldContainer<ScalarT> abscoeff;

 };
}

#endif
