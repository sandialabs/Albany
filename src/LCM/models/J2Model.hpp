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
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields);

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
public:

  struct have_temperature_Tag{};
  struct dont_have_temperature_Tag{};

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy<ExecutionSpace, have_temperature_Tag>  have_temperature_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, dont_have_temperature_Tag>  dont_have_temperature_Policy;

  void
  computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields);

  class computeStateKernel
  {

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
    PHX::MDField<ScalarT, Dim, Dim> F;
    PHX::MDField<ScalarT, Dim, Dim> be;
    PHX::MDField<ScalarT, Dim, Dim> s;
    PHX::MDField<ScalarT, Dim, Dim> sigma;
    PHX::MDField<ScalarT, Dim, Dim> N;
    PHX::MDField<ScalarT, Dim, Dim> A;
    PHX::MDField<ScalarT, Dim, Dim> expA;
    PHX::MDField<ScalarT, Dim, Dim> Fpnew;
    PHX::MDField<ScalarT, Dim, Dim> I;
    PHX::MDField<ScalarT, Dim, Dim> Fpn;
    PHX::MDField<ScalarT, Dim, Dim> Fpinv;
    PHX::MDField<ScalarT, Dim, Dim> Cpinv;

    ScalarT kappa, mu, mubar, K, Y;
    ScalarT Jm23, trace, smag2, smag, f, p, dgam;
    ScalarT sq23;

   public:
   typedef PHX::Device device_type;

    computeStateKernel ( int dims, 
                         int num_pts_,
                         ArrayT &def_grad_,
                         ArrayT &J_,
                         ArrayT &poissons_ratio_,
                         ArrayT &elastic_modulus_, 
                         ArrayT &yieldStrength_,
                         ArrayT &hardeningModulus_,
                         ArrayT &delta_time_,
                         ArrayT &stress_,
                         ArrayT &Fp_,
                         ArrayT &eqps_,
                         ArrayT &yieldSurf_,
                         ArrayT &source_,
                         Albany::MDArray &Fpold_,
                         Albany::MDArray &eqpsold_)
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
    {

     typedef PHX::KokkosViewFactory<ScalarT,PHX::Device> ViewFactory;
     std::vector<PHX::index_size_type> ddims_;
     ddims_.push_back(24);

      F     = PHX::MDField<ScalarT, Dim, Dim>("F",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      be    = PHX::MDField<ScalarT, Dim, Dim>("be",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      s     = PHX::MDField<ScalarT, Dim, Dim>("s",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      sigma = PHX::MDField<ScalarT, Dim, Dim>("sigma",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      N     = PHX::MDField<ScalarT, Dim, Dim>("N",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      A     = PHX::MDField<ScalarT, Dim, Dim>("A",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      expA  = PHX::MDField<ScalarT, Dim, Dim>("expA",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      Fpnew = PHX::MDField<ScalarT, Dim, Dim>("Fpnew",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      I     = PHX::MDField<ScalarT, Dim, Dim>("I",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      Fpn   = PHX::MDField<ScalarT, Dim, Dim>("Fpn",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      Fpinv = PHX::MDField<ScalarT, Dim, Dim>("Fpinv",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));
      Cpinv = PHX::MDField<ScalarT, Dim, Dim>("Cpinv",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(dims_,dims_)));

     F.setFieldData(ViewFactory::buildView(F.fieldTag(),ddims_));
     be.setFieldData(ViewFactory::buildView(be.fieldTag(),ddims_));
     s.setFieldData(ViewFactory::buildView(s.fieldTag(),ddims_));
     sigma.setFieldData(ViewFactory::buildView(sigma.fieldTag(),ddims_));
     N.setFieldData(ViewFactory::buildView(N.fieldTag(),ddims_));
     A.setFieldData(ViewFactory::buildView(A.fieldTag(),ddims_));
     expA.setFieldData(ViewFactory::buildView(expA.fieldTag(),ddims_));
     Fpnew.setFieldData(ViewFactory::buildView(Fpnew.fieldTag(),ddims_));
     I.setFieldData(ViewFactory::buildView(I.fieldTag(),ddims_));
     Fpn.setFieldData(ViewFactory::buildView(Fpn.fieldTag(),ddims_));
     Fpinv.setFieldData(ViewFactory::buildView(Fpinv.fieldTag(),ddims_));
     Cpinv.setFieldData(ViewFactory::buildView(Cpinv.fieldTag(),ddims_));

    sq23=(std::sqrt(2. / 3.));

    }

 /*   struct have_temperature_Tag{};
    struct dont_have_temperature_Tag{};
    
    typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

    typedef Kokkos::RangePolicy<ExecutionSpace, have_temperature_Tag>  have_temperature_Policy; 
    typedef Kokkos::RangePolicy<ExecutionSpace, dont_have_temperature_Tag>  dont_have_temperature_Policy;
*/
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

    template <class ArrayT>
    KOKKOS_INLINE_FUNCTION
    const ArrayT  transpose(const ArrayT& A) const;

    template <class ArrayT>
    KOKKOS_INLINE_FUNCTION
    const ArrayT  inverse(const ArrayT& A) const;  
 
  };
};
}

#endif
