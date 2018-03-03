//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ACEice_hpp)
#define LCM_ACEice_hpp

#include "../parallel_models/ParallelConstitutiveModel.hpp"

namespace LCM {

template<typename EvalT, typename Traits>
struct ACEiceMiniKernel : public ParallelKernel<EvalT, Traits>
{
  ///
  /// Constructor
  ///
  ACEiceMiniKernel(
      ConstitutiveModel<EvalT, Traits>& model,
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

  using BaseKernel::num_dims_;
  using BaseKernel::num_pts_;
  using BaseKernel::field_name_map_;

  // optional temperature support
  using BaseKernel::expansion_coeff_;
  using BaseKernel::have_temperature_;
  using BaseKernel::latent_heat_;
  using BaseKernel::ref_temperature_;
  using BaseKernel::temperature_;

  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;
  using BaseKernel::addStateVariable;

  /// Pointer to NOX status test, allows the material model to force
  /// a global load step reduction
  using BaseKernel::nox_status_test_;

  // Dependent MDFields
  ConstScalarField def_grad;
  ConstScalarField delta_time;
  ConstScalarField density;
  ConstScalarField elastic_modulus;
  ConstScalarField hardening_modulus;
  ConstScalarField heat_capacity;
  ConstScalarField J;
  ConstScalarField poissons_ratio;
  ConstScalarField thermal_conductivity;
  ConstScalarField yield_strength;


  // extract evaluated MDFields
  ScalarField stress;
  ScalarField Fp;
  ScalarField eqps;
  ScalarField yieldSurf;
  ScalarField source;

  Albany::MDArray Fpold;
  Albany::MDArray eqpsold;

  // Saturation hardening constraints
  RealType sat_mod;
  RealType sat_exp;

  void
  init(
      Workset&                 workset,
      FieldMap<ScalarT const>& dep_fields,
      FieldMap<ScalarT>&       eval_fields);

  KOKKOS_INLINE_FUNCTION
  void
  operator()(int cell, int pt) const;
};

template<typename EvalT, typename Traits>
class ACEice : public LCM::ParallelConstitutiveModel<
                         EvalT,
                         Traits,
                         ACEiceMiniKernel<EvalT, Traits>> {
 public:
  ACEice(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);
};
}
#endif  // LCM_ACEice_hpp
