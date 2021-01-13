//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_P0_INTERPOLATION_HPP
#define PHAL_P0_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_SacadoTypes.hpp"

#include "Intrepid2_CellTools.hpp"

namespace PHAL
{
/** \brief Average from points to cell/side

    This evaluator averages the node/quadpoints values to
    obtain a single value for the whole cell/side

*/

template<typename EvalT, typename Traits, typename ScalarT>
class P0InterpolationBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  P0InterpolationBase (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluate_on_side (typename Traits::EvalData d);
  void evaluate_on_cell (typename Traits::EvalData d);

  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename Albany::StrongestScalarType<ScalarT,MeshScalarT>::type OutputScalarT;

  enum InterpolationType {
    ValueAtCellBarycenter,
    CellAverage
  };

  bool eval_on_side;  // Whether the interpolation is on a volume or side field.

  int numQPs; 
  int numNodes;
  int dim0;     // For rank1 and rank2 fields
  int dim1;     // For rank2 fields

  Albany::FieldLocation loc;
  Albany::FieldRankType rank;
  InterpolationType  itype;

  std::vector<PHX::DataLayout::size_type> dims;

  std::string sideSetName; // Only if eval_on_side=true

  // These are only needed for barycenter interpolation
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;
  Kokkos::DynRankView<RealType, PHX::Device>                       basis_at_barycenter;

  MDFieldMemoizer<Traits> memoizer;

  // Input:
  PHX::MDField<const ScalarT>       field;
  PHX::MDField<const RealType>      BF;
  PHX::MDField<const MeshScalarT>   w_measure;

  // Output:
  PHX::MDField<OutputScalarT>       field_p0;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct Cell_Average_Scalar_Field_Tag{};
  struct Cell_Average_Vector_Field_Tag{};
  struct Cell_Average_Tensor_Field_Tag{};
  struct Cell_Barycenter_Scalar_Field_Tag{};
  struct Cell_Barycenter_Vector_Field_Tag{};
  struct Cell_Barycenter_Tensor_Field_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Average_Scalar_Field_Tag> Cell_Average_Scalar_Field_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Average_Vector_Field_Tag> Cell_Average_Vector_Field_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Average_Tensor_Field_Tag> Cell_Average_Tensor_Field_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Barycenter_Scalar_Field_Tag> Cell_Barycenter_Scalar_Field_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Barycenter_Vector_Field_Tag> Cell_Barycenter_Vector_Field_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,Cell_Barycenter_Tensor_Field_Tag> Cell_Barycenter_Tensor_Field_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Average_Scalar_Field_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Average_Vector_Field_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Average_Tensor_Field_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Barycenter_Scalar_Field_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Barycenter_Vector_Field_Tag& tag, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const Cell_Barycenter_Tensor_Field_Tag& tag, const int& cell) const;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using P0Interpolation = P0InterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using P0InterpolationMesh = P0InterpolationBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using P0InterpolationParam = P0InterpolationBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace PHAL

#endif // PHAL_P0_INTERPOLATION_HPP
