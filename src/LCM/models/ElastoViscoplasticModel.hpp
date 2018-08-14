//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ElastoViscoplasticModel_hpp)
#define LCM_ElastoViscoplasticModel_hpp

#include "ConstitutiveModel.hpp"
#include "ElastoViscoplasticCore.hpp"

namespace LCM {

//! \brief Elasto Viscoplastic Constitutive Model
template <typename EvalT, typename Traits>
class ElastoViscoplasticModel : public LCM::ConstitutiveModel<EvalT, Traits>
{
 public:
  using Base        = LCM::ConstitutiveModel<EvalT, Traits>;
  using DepFieldMap = typename Base::DepFieldMap;
  using FieldMap    = typename Base::FieldMap;

  typedef typename EvalT::ScalarT             ScalarT;
  typedef typename EvalT::MeshScalarT         MeshScalarT;
  typedef typename Sacado::Fad::DFad<ScalarT> Fad;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

  // optional temperature support
  using ConstitutiveModel<EvalT, Traits>::have_temperature_;
  using ConstitutiveModel<EvalT, Traits>::expansion_coeff_;
  using ConstitutiveModel<EvalT, Traits>::ref_temperature_;
  using ConstitutiveModel<EvalT, Traits>::heat_capacity_;
  using ConstitutiveModel<EvalT, Traits>::density_;
  using ConstitutiveModel<EvalT, Traits>::temperature_;
  using ConstitutiveModel<EvalT, Traits>::have_total_concentration_;
  using ConstitutiveModel<EvalT, Traits>::have_total_bubble_density_;
  using ConstitutiveModel<EvalT, Traits>::have_bubble_volume_fraction_;
  using ConstitutiveModel<EvalT, Traits>::total_concentration_;
  using ConstitutiveModel<EvalT, Traits>::total_bubble_density_;
  using ConstitutiveModel<EvalT, Traits>::bubble_volume_fraction_;

  ///
  /// Constructor
  ///
  ElastoViscoplasticModel(
      Teuchos::ParameterList*              p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Denstructor
  ///
  virtual ~ElastoViscoplasticModel(){};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual void
  computeState(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields);

  virtual void
  computeStateParallel(
      typename Traits::EvalData workset,
      DepFieldMap               dep_fields,
      FieldMap                  eval_fields)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
  }

 private:
  ///
  /// Private to prohibit copying
  ///
  ElastoViscoplasticModel(const ElastoViscoplasticModel&);

  ///
  /// Private to prohibit copying
  ///
  ElastoViscoplasticModel&
  operator=(const ElastoViscoplasticModel&);

  ///
  /// compute stresses
  ///
  // template<typename ArgT>
  // void
  // computeStress(minitensor::Tensor<ScalarT> const & F,
  //     minitensor::Tensor<ArgT> const & Fp,
  //     minitensor::Tensor<ArgT> & T,
  //     minitensor::Tensor<ArgT> & S,
  //     std::vector<ArgT> & shear) const;

  ///
  /// Initial Void Volume
  /// xs
  RealType f0_;

  ///
  /// Shear Damage Parameter
  ///
  RealType kw_;

  ///
  /// Void Nucleation Parameters
  ///
  RealType eN_, sN_, fN_;

  ///
  /// Void Nucleation Parameters with H, He
  ///
  RealType eHN_, eHN_coeff_, sHN_, fHeN_, fHeN_coeff_;

  ///
  /// Critical Void Parameters
  ///
  RealType fc_, ff_;

  ///
  /// Gurson yield surface parameters
  ///
  RealType q1_, q2_, q3_;

  ///
  /// Hydrogen and Helium yield surface parameters
  ///
  RealType alpha1_, alpha2_, Ra_;

  ///
  /// Flow Rule Scale Factor
  ///
  RealType f_scale_;

  ///
  /// Solution options
  ///
  bool                 apply_slip_predictor_;
  minitensor::StepType step_type_;

  RealType implicit_nonlinear_solver_relative_tolerance_;
  RealType implicit_nonlinear_solver_absolute_tolerance_;
  int      implicit_nonlinear_solver_max_iterations_;
  int      implicit_nonlinear_solver_min_iterations_;

  ///
  /// flag to print convergence
  ///
  bool print_;

  ///
  /// Compute effective void volume fraction
  ///
  template <typename T>
  T
  compute_fstar(T f, RealType fc, RealType ff, RealType q1);
};
}  // namespace LCM

#endif
