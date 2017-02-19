//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_J2MiniSolver_hpp)
#define LCM_J2MiniSolver_hpp

#include "../parallel_models/ParallelConstitutiveModel.hpp"

namespace LCM
{
template<typename EvalT, typename Traits>
struct J2MiniKernel : public ParallelKernel<EvalT, Traits>
{
  ///
  /// Constructor
  ///
  J2MiniKernel(
      ConstitutiveModel<EvalT, Traits> &model,
      Teuchos::ParameterList * p,
      Teuchos::RCP<Albany::Layouts> const & dl);

  ///
  /// No copy constructor
  ///
  J2MiniKernel(J2MiniKernel const &) = delete;

  ///
  /// No copy assignment
  ///
  J2MiniKernel & operator=(J2MiniKernel const &) = delete;

  using ScalarT = typename EvalT::ScalarT;
  using ScalarField = PHX::MDField<ScalarT>;
  using ConstScalarField = PHX::MDField<const ScalarT>;
  using BaseKernel = ParallelKernel<EvalT, Traits>;
  using Workset = typename BaseKernel::Workset;

  using BaseKernel::num_dims_;
  using BaseKernel::num_pts_;
  using BaseKernel::field_name_map_;

  // optional temperature support
  using BaseKernel::have_temperature_;
  using BaseKernel::expansion_coeff_;
  using BaseKernel::ref_temperature_;
  using BaseKernel::heat_capacity_;
  using BaseKernel::density_;
  using BaseKernel::temperature_;
  
  using BaseKernel::setDependentField;
  using BaseKernel::setEvaluatedField;
  using BaseKernel::addStateVariable;
  
  // Dependent MDFields
  ConstScalarField def_grad;
  ConstScalarField J;
  ConstScalarField poissons_ratio;
  ConstScalarField elastic_modulus;
  ConstScalarField yieldStrength;
  ConstScalarField hardeningModulus;
  ConstScalarField delta_time;

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
  init(Workset &workset,
       FieldMap<const ScalarT> &dep_fields,
       FieldMap<ScalarT> &eval_fields);

  KOKKOS_INLINE_FUNCTION
  void operator() (int cell, int pt) const;
};

template<typename EvalT, typename Traits>
class J2MiniSolver : public LCM::ParallelConstitutiveModel<EvalT, Traits, J2MiniKernel<EvalT, Traits>> {
public:
  J2MiniSolver(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);
};

}
#endif // LCM_J2MiniSolver_hpp
