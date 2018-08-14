//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(HMC_TotalStress_hpp)
#define HMC_TotalStress_hpp

#include "Albany_Layouts.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace HMC {
template <typename EvalT, typename Traits>
class TotalStress : public PHX::EvaluatorWithBaseImpl<Traits>,
                    public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ///
  /// Constructor
  ///
  TotalStress(
      const Teuchos::ParameterList&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Implementation of physics
  ///
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT                                ScalarT;
  typedef typename EvalT::MeshScalarT                            MeshScalarT;
  typedef PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> cHMC2Tensor;

  ///
  /// Input: macro stress and micro stresses
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> macroStress;
  Teuchos::ArrayRCP<Teuchos::RCP<cHMC2Tensor>>           microStress;

  ///
  /// Output: total stress
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> totalStress;

  ///
  /// Number of integration points
  ///
  unsigned int numQPs;

  ///
  /// Number of problem dimensions
  ///
  unsigned int numDims;

  ///
  /// Number of micro scales
  ///
  unsigned int numMicroScales;
};
}  // namespace HMC

#endif
