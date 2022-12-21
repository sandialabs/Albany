//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint> Neutron;
  PHX::MDField<const ScalarT,Cell,QuadPoint> NeutronDiff;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> NGrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint> Absorp;
  PHX::MDField<const ScalarT,Cell,QuadPoint> Fission;  
  PHX::MDField<const ScalarT,Cell,QuadPoint> nu;  
  
  PHX::MDField<const ScalarT,Cell,QuadPoint> Source;

  // Output:
  PHX::MDField<ScalarT,Cell,Node> NResidual;

  bool haveNeutSource;
  unsigned int numQPs, numDims, numNodes, numCells;
  Kokkos::DynRankView<ScalarT, PHX::Device> flux;
  Kokkos::DynRankView<ScalarT, PHX::Device> abscoeff;

 };
}

#endif
