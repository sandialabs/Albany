//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ParallelNeohookeanModel_hpp)
#define LCM_ParallelNeohookeanModel_hpp

#include "Albany_Layouts.hpp"
#include "ParallelConstitutiveModel.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

namespace LCM {
template <typename EvalT, typename Traits>
struct NeohookeanKernel : public ParallelKernel<EvalT, Traits>
{
  ///
  /// Constructor
  ///
  NeohookeanKernel(
      ConstitutiveModel<EvalT, Traits>&    model,
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  NeohookeanKernel(const NeohookeanKernel&) = delete;
  NeohookeanKernel&
  operator=(const NeohookeanKernel&) = delete;

  using ScalarT          = typename EvalT::ScalarT;
  using MeshScalarT      = typename EvalT::MeshScalarT;
  using ScalarField      = PHX::MDField<ScalarT>;
  using ConstScalarField = PHX::MDField<const ScalarT>;
  using BaseKernel       = ParallelKernel<EvalT, Traits>;
  using Workset          = typename BaseKernel::Workset;

  using BaseKernel::compute_energy_;
  using BaseKernel::compute_tangent_;
  using BaseKernel::field_name_map_;
  using BaseKernel::num_dims_;
  using BaseKernel::num_pts_;

  using BaseKernel::addStateVariable;
  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;

  using BaseKernel::nox_status_test_;

  // Dependent MDFields
  ConstScalarField def_grad;
  ConstScalarField J;
  ConstScalarField poissons_ratio;
  ConstScalarField elastic_modulus;

  // Evaluated MDFields
  ScalarField stress;
  ScalarField energy;
  ScalarField tangent;

  void
  init(
      Workset&                 workset,
      FieldMap<const ScalarT>& dep_fields,
      FieldMap<ScalarT>&       eval_fields);

  KOKKOS_INLINE_FUNCTION
  void
  operator()(int cell, int pt) const;
};

template <typename EvalT, typename Traits>
class ParallelNeohookeanModel : public LCM::ParallelConstitutiveModel<
                                    EvalT,
                                    Traits,
                                    NeohookeanKernel<EvalT, Traits>>
{
 public:
  ParallelNeohookeanModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);
};

#if 0
//! \brief Parallel Neohookean Model
template<typename EvalT, typename Traits>
class ParallelNeohookeanModel: public LCM::ParallelConstitutiveModel<EvalT, Traits, detail::NeohookeanKernel<EvalT, Traits>>
{
public:

  using Parent = ParallelConstitutiveModel<EvalT, Traits, detail::NeohookeanKernel<EvalT, Traits>>;
  using ScalarT = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;

  using FieldMap = typename Parent::FieldMap;
  using EvalKernel = typename Parent::EvalKernel;

  using Parent::num_dims_;
  using Parent::num_pts_;
  using Parent::field_name_map_;
  using Parent::compute_energy_;
  using Parent::compute_tangent_;

  // optional temperature support
  using Parent::have_temperature_;
  using Parent::expansion_coeff_;
  using Parent::ref_temperature_;
  using Parent::heat_capacity_;
  using Parent::density_;
  using Parent::temperature_;

  ///
  /// Virtual Destructor
  ///
  virtual
  ~ParallelNeohookeanModel() = default;

protected:

  virtual
  EvalKernel
  createEvalKernel(typename Traits::EvalData &workset,
                   FieldMap &dep_fields,
                   FieldMap &eval_fields) override;

};
#endif
}  // namespace LCM

#endif
