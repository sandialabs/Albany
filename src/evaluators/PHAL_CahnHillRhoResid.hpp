//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_CAHNHILLRHORESID_HPP
#define PHAL_CAHNHILLRHORESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class CahnHillRhoResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  typedef typename EvalT::ScalarT ScalarT;

  CahnHillRhoResid(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<const ScalarT,Cell,QuadPoint,Dim> rhoGrad;
  PHX::MDField<const ScalarT,Cell,QuadPoint> chemTerm;
  PHX::MDField<const ScalarT,Cell,QuadPoint> noiseTerm;


  // Output:
  PHX::MDField<ScalarT,Cell,Node> rhoResidual;

  Kokkos::DynRankView<ScalarT, PHX::Device> gamma_term;

  unsigned int numQPs, numDims, numNodes, worksetSize;

  ScalarT gamma;

  // Langevin noise present
  bool haveNoise;


};
}

#endif
