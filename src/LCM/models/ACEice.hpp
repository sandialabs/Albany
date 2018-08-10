//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ACEice_hpp)
#define LCM_ACEice_hpp

#include "ParallelConstitutiveModel.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
struct ACEiceMiniKernel : public ParallelKernel<EvalT, Traits>
{
  ///
  /// Constructor
  ///
  ACEiceMiniKernel(
      ConstitutiveModel<EvalT, Traits>&    model,
      Teuchos::ParameterList*              p,
      Teuchos::RCP<Albany::Layouts> const& dl);

  ///
  /// No copy constructor
  ///
  ACEiceMiniKernel(ACEiceMiniKernel const&) = delete;

  ///
  /// No copy assignment
  ///
  ACEiceMiniKernel&
  operator=(ACEiceMiniKernel const&) = delete;

  using ScalarT          = typename EvalT::ScalarT;
  using ScalarField      = PHX::MDField<ScalarT>;
  using ConstScalarField = PHX::MDField<ScalarT const>;
  using BaseKernel       = ParallelKernel<EvalT, Traits>;
  using Workset          = typename BaseKernel::Workset;

  using BaseKernel::field_name_map_;
  using BaseKernel::num_dims_;
  using BaseKernel::num_pts_;

  // optional temperature support
  using BaseKernel::expansion_coeff_;
  using BaseKernel::have_temperature_;
  using BaseKernel::ref_temperature_;

  using BaseKernel::addStateVariable;
  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;

  /// Pointer to NOX status test, allows the material model to force
  /// a global load step reduction
  using BaseKernel::nox_status_test_;

  // Input constant MDFields
  ConstScalarField def_grad_;
  ConstScalarField delta_time_;
  ConstScalarField elastic_modulus_;
  ConstScalarField hardening_modulus_;
  ConstScalarField J_;
  ConstScalarField poissons_ratio_;
  ConstScalarField yield_strength_;
  ConstScalarField temperature_;

  // Output MDFields
  ScalarField density_;
  ScalarField heat_capacity_;
  ScalarField ice_saturation_;
  ScalarField thermal_cond_;
  ScalarField thermal_inertia_;
  ScalarField water_saturation_;
  ScalarField porosity_;
  ScalarField tdot_;

  // Mechanical MDFields
  ScalarField eqps_;
  ScalarField Fp_;
  ScalarField source_;
  ScalarField stress_;
  ScalarField yield_surf_;

  // Workspace arrays
  Albany::MDArray Fp_old_;
  Albany::MDArray eqps_old_;
  Albany::MDArray T_old_;
  Albany::MDArray ice_saturation_old_;

  // Baseline constants
  RealType ice_density_{0.0};
  RealType water_density_{0.0};
  RealType ice_thermal_cond_{0.0};
  RealType water_thermal_cond_{0.0};
  RealType ice_heat_capacity_{0.0};
  RealType water_heat_capacity_{0.0};
  RealType ice_saturation_init_{0.0};
  RealType ice_saturation_max_{0.0};
  RealType water_saturation_min_{0.0};
  RealType latent_heat_{0.0};
  RealType porosity0_{0.0};
  RealType porosityE_{0.0};
  RealType T_init_{0.0};

  // Saturation hardening constraints
  RealType sat_mod_{0.0};
  RealType sat_exp_{0.0};

  void
  init(
      Workset&                 workset,
      FieldMap<ScalarT const>& input_fields,
      FieldMap<ScalarT>&       output_fields);

  KOKKOS_INLINE_FUNCTION
  void
  operator()(int cell, int pt) const;
};

template <typename EvalT, typename Traits>
class ACEice : public LCM::ParallelConstitutiveModel<
                   EvalT,
                   Traits,
                   ACEiceMiniKernel<EvalT, Traits>>
{
 public:
  ACEice(Teuchos::ParameterList* p, const Teuchos::RCP<Albany::Layouts>& dl);
};
}  // namespace LCM
#endif  // LCM_ACEice_hpp
