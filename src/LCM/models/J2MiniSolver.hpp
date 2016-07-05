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

template<typename S>
using FieldMap = std::map<std::string, Teuchos::RCP<PHX::MDField<S>>>;

namespace detail {
  template<typename EvalT, typename Traits>
  struct J2MiniKernel
  {
    using ScalarT = typename EvalT::ScalarT;
    using ScalarField = PHX::MDField<ScalarT>;
    // Dependent MDFields
    ScalarField def_grad;
    ScalarField J;
    ScalarField poissons_ratio;
    ScalarField elastic_modulus;
    ScalarField yieldStrength;
    ScalarField hardeningModulus;
    ScalarField delta_time;

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
    
    // Temperature support
    bool have_temperature;
    PHX::MDField<ScalarT, Cell, QuadPoint> temperature;
    RealType density;
    RealType heat_capacity;
    
    // Misc parameters
    int num_dims, num_pts;
    bool compute_energy;
    bool compute_tangent;
    
    KOKKOS_INLINE_FUNCTION
    void operator() (int cell) const;
  };
}

///
/// J2 Plasticity Constitutive Model
///
template<typename EvalT, typename Traits>
class J2MiniSolver: public LCM::ParallelConstitutiveModel<EvalT, Traits, detail::J2MiniKernel<EvalT, Traits> >
{
public:
  
  using BaseClass = ParallelConstitutiveModel<EvalT, Traits, detail::J2MiniKernel<EvalT, Traits> >;
  using ScalarT = typename EvalT::ScalarT;
  using MeshScalarT = typename EvalT::MeshScalarT;
  using EvalKernel = typename BaseClass::EvalKernel;
  using FieldMap = typename BaseClass::FieldMap;

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

  ///
  /// Constructor
  ///
  J2MiniSolver(
      Teuchos::ParameterList * p,
      Teuchos::RCP<Albany::Layouts> const & dl);

  ///
  /// No copy constructor
  ///
  J2MiniSolver(J2MiniSolver const &) = delete;

  ///
  /// No copy assignment
  ///
  J2MiniSolver & operator=(J2MiniSolver const &) = delete;

  ///
  /// Virtual Denstructor
  ///
  virtual
  ~J2MiniSolver()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  /*virtual
  void
  computeState(
      typename Traits::EvalData workset,
      FieldMap<ScalarT> dep_fields,
      FieldMap<ScalarT> eval_fields);*/

  //
  // This is required by Kokkos.
  //

  // Where the threads will execute (serial, openmp, pthreads, CUDA, etc.)
  using ExecutionSpace = Kokkos::View<int***, PHX::Device>::execution_space;

  // This tag is just a label that is useful in selecting execution policies
  // and functors by using it as a template parameter to instantiate
  // the desired policy or functor.
  struct J2ResidualTag {};

  // Functors that use this policy will be executed over a range
  // of an integral type. Adequate for parallel execution on an index,
  // normally cell index.
  using J2ResidualPolicy =
      Kokkos::RangePolicy<ExecutionSpace, J2ResidualTag>;

  // This functor computes the J2 residual in parallel.
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const J2ResidualTag & tag, int const & cell) const;
  
protected:
  
  virtual
  EvalKernel
  createEvalKernel(typename Traits::EvalData &workset,
                   FieldMap &dep_fields,
                   FieldMap &eval_fields) override;

private:

  ///
  /// Saturation hardening constants
  ///
  RealType
  sat_mod_;

  RealType
  sat_exp_;
#if 0
  //Kokkos
  virtual
  void
  computeStateParallel(
      typename Traits::EvalData workset,
      FieldMap<ScalarT> dep_fields,
      FieldMap<ScalarT> eval_fields);
#endif
};

}
#endif // LCM_J2MiniSolver_hpp
