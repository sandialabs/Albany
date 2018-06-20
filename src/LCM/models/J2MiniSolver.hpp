//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_J2MiniSolver_hpp)
#define LCM_J2MiniSolver_hpp

#include "ParallelConstitutiveModel.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
struct J2MiniKernel : public ParallelKernel<EvalT, Traits>
{
  ///
  /// Constructor
  ///
  J2MiniKernel(
      ConstitutiveModel<EvalT, Traits>&    model,
      Teuchos::ParameterList*              p,
      Teuchos::RCP<Albany::Layouts> const& dl);

  ///
  /// No copy constructor
  ///
  J2MiniKernel(J2MiniKernel const&) = delete;

  ///
  /// No copy assignment
  ///
  J2MiniKernel&
  operator=(J2MiniKernel const&) = delete;

  using ScalarT          = typename EvalT::ScalarT;
  using ScalarField      = PHX::MDField<ScalarT>;
  using ConstScalarField = PHX::MDField<ScalarT const>;
  using BaseKernel       = ParallelKernel<EvalT, Traits>;
  using Workset          = typename BaseKernel::Workset;

  using BaseKernel::field_name_map_;
  using BaseKernel::num_dims_;
  using BaseKernel::num_pts_;

  // optional temperature support
  using BaseKernel::density_;
  using BaseKernel::expansion_coeff_;
  using BaseKernel::have_temperature_;
  using BaseKernel::heat_capacity_;
  using BaseKernel::ref_temperature_;

  using BaseKernel::addStateVariable;
  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;

  /// Pointer to NOX status test, allows the material model to force
  /// a global load step reduction
  using BaseKernel::nox_status_test_;

  // Dependent MDFields
  ConstScalarField def_grad_;
  ConstScalarField delta_time_;
  ConstScalarField elastic_modulus_;
  ConstScalarField hardening_modulus_;
  ConstScalarField J_;
  ConstScalarField poissons_ratio_;
  ConstScalarField yield_strength_;
  ConstScalarField temperature_;

  // extract evaluated MDFields
  ScalarField eqps_;
  ScalarField Fp_;
  ScalarField source_;
  ScalarField stress_;
  ScalarField yield_surf_;

  Albany::MDArray Fp_old_;
  Albany::MDArray eqps_old_;

  // Saturation hardening constraints
  RealType sat_mod_;
  RealType sat_exp_;

  void
  init(
      Workset&                 workset,
      FieldMap<ScalarT const>& dep_fields,
      FieldMap<ScalarT>&       eval_fields);

  KOKKOS_INLINE_FUNCTION
  void
  operator()(int cell, int pt) const;
};

template <typename EvalT, typename Traits>
class J2MiniSolver
    : public LCM::
          ParallelConstitutiveModel<EvalT, Traits, J2MiniKernel<EvalT, Traits>>
{
 public:
  J2MiniSolver(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);
};
}  // namespace LCM
#endif  // LCM_J2MiniSolver_hpp
