//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(HMC_StrainDifference_hpp)
#define HMC_StrainDifference_hpp

#include "Albany_Layouts.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace HMC {
template <typename EvalT, typename Traits>
class StrainDifference : public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ///
  /// Constructor
  ///
  StrainDifference(
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
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ///
  /// Input: displacement gradient
  ///
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> macroStrain;
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim> microStrain;

  ///
  /// Output: strainDifference
  ///
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> strainDifference;

  ///
  /// Number of integration points
  ///
  unsigned int numQPs;

  ///
  /// Number of problem dimensions
  ///
  unsigned int numDims;
};
}  // namespace HMC

#endif
