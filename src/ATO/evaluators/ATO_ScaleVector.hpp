//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_SCALE_VECTOR_HPP
#define ATO_SCALE_VECTOR_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace ATO {
/** \brief Computes outVector = coefficient*inVector

    This evaluator scales a vector.

*/

template<typename EvalT, typename Traits>
class ScaleVector : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ScaleVector(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> inVector;
  double coefficient;

  unsigned int numQPs;
  unsigned int numDims;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> outVector;

  bool addCellForcing;
  bool useHomogenizedConstants;
  std::string homogenizedConstantsName;
  int cellForcingColumn;

  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> subTensor;
};
}

#endif
