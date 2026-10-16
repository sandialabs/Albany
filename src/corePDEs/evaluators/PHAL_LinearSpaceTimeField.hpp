//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LINEAR_SPACE_TIME_FIELD_HPP
#define PHAL_LINEAR_SPACE_TIME_FIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Kokkos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"

namespace PHAL {

/** \brief Field that is linear in space and time, at the quadrature points:

      f(x,t) = c_0 + c_1 x_1 + ... + c_d x_d + c_{d+1} t,

    with d the number of spatial dimensions (at most 3) and t the workset time.
    The coefficients are given in the parameter "Coefficients", as [c_0, c_1, ..., c_{d+1}].
    The field only depends on the coordinates and on time, hence its type is MeshScalarT.

    Example: the target temperature of the transient optimization tests of the Heat problem
    (tests/corePDEs/TransientHeat2D), used as the "Target Field Name" of the response
    "Squared L2 Difference Source ST Target MST".
*/
template<typename EvalT, typename Traits>
class LinearSpaceTimeField : public PHX::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  typedef typename EvalT::MeshScalarT MeshScalarT;

  LinearSpaceTimeField (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields (typename Traits::EvalData d);

  // Kokkos functor
  struct LinearSpaceTimeField_Tag{};
  typedef PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, LinearSpaceTimeField_Tag> LinearSpaceTimeField_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const LinearSpaceTimeField_Tag& tag, const int& cell) const;

private:
  // Input:
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;

  // Output:
  PHX::MDField<MeshScalarT,Cell,QuadPoint> field;

  int numQPs;
  int numDims;

  // Spatial coefficients [c_1, ..., c_d] (unused entries are zero), and c_0 + c_{d+1} t
  Kokkos::Array<double,3> coeffs_x;
  double c_0;
  double c_t;
  double value_at_origin;
};

} // namespace PHAL

#endif // PHAL_LINEAR_SPACE_TIME_FIELD_HPP
