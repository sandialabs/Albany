//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef STRESSES_HPP
#define STRESSES_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace HMC {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template <typename EvalT, typename Traits>
class Stresses : public PHX::EvaluatorWithBaseImpl<Traits>,
                 public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  Stresses(const Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  typedef PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> cHMC2Tensor;
  typedef PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim>       HMC2Tensor;
  typedef PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim, Dim>
                                                                cHMC3Tensor;
  typedef PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim, Dim> HMC3Tensor;
  typedef PHX::MDField<ScalarT, Cell, QuadPoint>                HMCScalar;

  // Input:
  cHMC2Tensor strain;

  Teuchos::ArrayRCP<Teuchos::RCP<cHMC2Tensor>> strainDifference;
  Teuchos::ArrayRCP<Teuchos::RCP<cHMC3Tensor>> microStrainGradient;
  std::vector<RealType>                        lengthScale;
  std::vector<RealType>                        betaParameter;

  RealType C11, C33, C12, C23, C44, C66;

  unsigned int numQPs;
  unsigned int numDims;
  unsigned int numMicroScales;

  // Output:
  HMC2Tensor stress;

  Teuchos::ArrayRCP<Teuchos::RCP<HMC2Tensor>> microStress;
  Teuchos::ArrayRCP<Teuchos::RCP<HMC3Tensor>> doubleStress;
};
}  // namespace HMC

#endif
