//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef HMC_ELASTICITYRESID_HPP
#define HMC_ELASTICITYRESID_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace HMC {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template <typename EvalT, typename Traits>
class MicroResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  MicroResidual(const Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim>      microStress;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim, Dim> doubleStress;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint, Dim> wGradBF;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>      wBF;

  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> epsDotDot;

  // Output:
  PHX::MDField<ScalarT, Cell, Node, Dim, Dim> ExResidual;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  bool        enableTransient;
};
}  // namespace HMC

#endif
