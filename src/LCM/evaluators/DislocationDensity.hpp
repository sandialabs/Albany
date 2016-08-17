//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef DISLOCATIONDENSITY_HPP
#define DISLOCATIONDENSITY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

#include "Teuchos_SerialDenseMatrix.hpp" 
#include "Teuchos_SerialDenseSolver.hpp"

namespace LCM {
/** \brief Dislocation Density Tensor

    This evaluator calculates the dislcation density tensor

*/

template<typename EvalT, typename Traits>
class DislocationDensity : public PHX::EvaluatorWithBaseImpl<Traits>,
			   public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  DislocationDensity(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  int  numVertices, numDims, numNodes, numQPs;
  bool square;

  // Input:
  PHX::MDField<double,Cell,QuadPoint,Dim,Dim> Fp;
  PHX::MDField<double,Cell,Node,QuadPoint> BF;
  PHX::MDField<double,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  PHX::MDField<double,Cell,QuadPoint,Dim,Dim> G;

  // Temporary Views
  Kokkos::DynRankView<RealType, PHX::Device> nodalFp;
  Kokkos::DynRankView<RealType, PHX::Device> curlFp;

};
}

#endif
