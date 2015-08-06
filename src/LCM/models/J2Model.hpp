//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_J2Model_hpp)
#define LCM_J2Model_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"

namespace LCM
{

//! \brief J2 Plasticity Constitutive Model
template<typename EvalT, typename Traits>
class J2Model: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

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
  J2Model(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Denstructor
  ///
  virtual
  ~J2Model()
  {};

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > eval_fields);

private:

  ///
  /// Private to prohibit copying
  ///
  J2Model(const J2Model&);

  ///
  /// Private to prohibit copying
  ///
  J2Model& operator=(const J2Model&);

  ///
  /// Saturation hardening constants
  ///
  RealType sat_mod_, sat_exp_;

 //Kokkos 
  virtual
  void
  computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>> > eval_fields);
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

  struct have_temperature_Tag{};
  struct dont_have_temperature_Tag{};

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy<ExecutionSpace, have_temperature_Tag>  have_temperature_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, dont_have_temperature_Tag>  dont_have_temperature_Policy;

  class computeStateKernel
  {

    int derivative_dim ;

   
    typedef typename PHX::MDField<ScalarT> ArrayT;
    
    PHX::MDField<ScalarT> def_grad;
    PHX::MDField<ScalarT> J;
    PHX::MDField<ScalarT> poissons_ratio;
    PHX::MDField<ScalarT> elastic_modulus;
    PHX::MDField<ScalarT> yieldStrength;
    PHX::MDField<ScalarT> hardeningModulus;
    PHX::MDField<ScalarT> delta_time;

    ArrayT stress, Fp, eqps, yieldSurf, source;
    Albany::MDArray Fpold;
    Albany::MDArray eqpsold;


    int dims_, num_pts;
/*    PHX::MDField<ScalarT, Cell, Dim, Dim> F;
    PHX::MDField<ScalarT, Cell, Dim, Dim> be;
    PHX::MDField<ScalarT, Cell, Dim, Dim> s;
    PHX::MDField<ScalarT, Cell, Dim, Dim> sigma;
    PHX::MDField<ScalarT, Cell, Dim, Dim> N;
    PHX::MDField<ScalarT, Cell, Dim, Dim> A;
    PHX::MDField<ScalarT, Cell, Dim, Dim> expA;
    PHX::MDField<ScalarT, Cell, Dim, Dim> Fpnew;
    PHX::MDField<ScalarT, Cell, Dim, Dim> I;
    PHX::MDField<ScalarT, Cell, Dim, Dim> Fpn;
    PHX::MDField<ScalarT, Cell, Dim, Dim> Fpinv;
    PHX::MDField<ScalarT, Cell, Dim, Dim> Cpinv;
*/
    ScalarT sq23;
    bool have_temperature_;

    RealType sat_mod_, sat_exp_;
    RealType heat_capacity_, density_;
    PHX::MDField<ScalarT, Cell, QuadPoint> temperature_;
    RealType ref_temperature_;
    RealType expansion_coeff_;

   public:
   typedef PHX::Device device_type;

    computeStateKernel (const int dims, 
                        const int num_pts_,
                        const ArrayT &def_grad_,
                        const ArrayT &J_,
                        const ArrayT &poissons_ratio_,
                        const ArrayT &elastic_modulus_, 
                        const ArrayT &yieldStrength_,
                        const ArrayT &hardeningModulus_,
                        const ArrayT &delta_time_,
                        ArrayT &stress_,
                        ArrayT &Fp_,
                        ArrayT &eqps_,
                        ArrayT &yieldSurf_,
                        ArrayT &source_,
                        const Albany::MDArray &Fpold_,
                        const Albany::MDArray &eqpsold_,
                        const bool have_temperature, 
/*                        PHX::MDField<ScalarT, Cell, Dim, Dim> &F_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &be_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &s_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &sigma_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &N_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &A_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &expA_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &Fpnew_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &I_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &Fpn_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &Fpinv_,
                        PHX::MDField<ScalarT, Cell, Dim, Dim> &Cpinv_,
*/                        const RealType sat_mod,
                        const RealType sat_exp,
                        const RealType heat_capacity,
                        const RealType density,
                        const PHX::MDField<ScalarT, Cell, QuadPoint> temperature,
                        const RealType ref_temperature,
                        const RealType expansion_coeff)
       : dims_(dims) 
       , num_pts(num_pts_)
       , def_grad(def_grad_)
       , J(J_)
       , poissons_ratio(poissons_ratio_)
       , elastic_modulus(elastic_modulus_)
       , yieldStrength(yieldStrength_)
       , hardeningModulus(hardeningModulus_)
       , delta_time(delta_time_)
       , stress(stress_)
       , Fp(Fp_)
       , eqps(eqps_)
       , yieldSurf(yieldSurf_)
       , source(source_)
       , Fpold(Fpold_)
       , eqpsold(eqpsold_)
       , have_temperature_(have_temperature)
/*       , F(F_)
       , be(be_)
       , s(s_)
       , sigma(sigma_)
       , N(N_)
       , A(A_)
       , expA(expA_)
       , Fpnew(Fpnew_)
       , I(I_)
       , Fpn(Fpn_)
       , Fpinv(Fpinv_)
       , Cpinv(Cpinv_)
*/       , sat_mod_(sat_mod)
       , sat_exp_(sat_exp)
       , heat_capacity_(heat_capacity)
       , density_(density)
       , temperature_(temperature)
       , ref_temperature_(ref_temperature)
       , expansion_coeff_(expansion_coeff)
    {

    sq23=(std::sqrt(2. / 3.));
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const have_temperature_Tag& tag, const int i) const;   

    KOKKOS_INLINE_FUNCTION
    void operator() (const dont_have_temperature_Tag& tag, const int i) const;

    KOKKOS_INLINE_FUNCTION
     void compute_common(const int cell) const;
    KOKKOS_INLINE_FUNCTION
     void compute_with_temperature(const int cell) const;
    KOKKOS_INLINE_FUNCTION
     void compute_with_no_temperature(const int cell) const;

    typedef  ScalarT TensorType [3][3] ; 
 
 /*  KOKKOS_INLINE_FUNCTION
   const ScalarT trace(const TensorType &A, const int dim) const;

   KOKKOS_INLINE_FUNCTION
   const ScalarT norm( const TensorType &A, const int dim) const;

   KOKKOS_INLINE_FUNCTION
   const ScalarT det(const TensorType &A, const int dim) const;
 */
  };

#endif  
};
}

#endif
