//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_CrystalPlasticityModel_hpp)
#define LCM_CrystalPlasticityModel_hpp

#include "CrystalPlasticityCore.hpp"
#include "ConstitutiveModel.hpp"

namespace LCM
{

//! \brief CrystalPlasticity Plasticity Constitutive Model
template<typename EvalT, typename Traits>
class CrystalPlasticityModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  enum class IntegrationScheme
  {
    UNDEFINED = 0, EXPLICIT = 1, IMPLICIT = 2
  };

  enum class ResidualType
  {
    UNDEFINED = 0, SLIP = 1, SLIP_HARDNESS = 2
  };

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // typedef for automatic differentiation type used in internal Newton loop
  // options are:  DFad (dynamically sized), SFad (static size), SLFad (bounded)
  // typedef typename Sacado::Fad::DFad<ScalarT> Fad;
  // typedef typename Sacado::Fad::SFad<ScalarT, CP::MAX_SLIP> Fad;
  typedef typename Sacado::Fad::SLFad<ScalarT, CP::MAX_SLIP> Fad;

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
  CrystalPlasticityModel(Teuchos::ParameterList* p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Virtual Denstructor
  ///
  virtual
  ~CrystalPlasticityModel()
  {
  }

  ///
  /// Method to compute the state (e.g. energy, stress, tangent)
  ///
  virtual
  void
  computeState(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>>dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields);

      virtual
      void
      computeStateParallel(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not implemented.");
      }

    private:

      ///
      /// Private to prohibit copying
      ///
      CrystalPlasticityModel(const CrystalPlasticityModel&);

      ///
      /// Private to prohibit copying
      ///
      CrystalPlasticityModel& operator=(const CrystalPlasticityModel&);

      ///
      /// Crystal elasticity parameters
      ///
      RealType c11_, c12_, c44_;
      Intrepid2::Tensor4<RealType, CP::MAX_DIM> C_;
      Intrepid2::Tensor<RealType, CP::MAX_DIM> orientation_;

      ///
      /// Number of slip systems
      ///
      int num_slip_;

      ///
      /// Crystal Plasticity parameters
      ///
      RealType rate_slip_reference_, exponent_rate_, energy_activation_, 
        H_, Rd_, tau_critical_,
        resistance_slip_initial_, rate_hardening_, stress_saturation_initial_,
        exponent_saturation_;

      std::vector< CP::SlipSystemStruct<CP::MAX_DIM,CP::MAX_SLIP> > 
      slip_systems_;

      IntegrationScheme integration_scheme_;
      ResidualType residual_type_;
      Intrepid2::StepType step_type_;
      FlowRule flow_rule_;
      HardeningLaw hardening_law_;
      RealType implicit_nonlinear_solver_relative_tolerance_;
      RealType implicit_nonlinear_solver_absolute_tolerance_;
      int implicit_nonlinear_solver_max_iterations_;
      int implicit_nonlinear_solver_min_iterations_;
      bool apply_slip_predictor_;
      int verbosity_;
      bool write_data_file_;
    };
  }

#endif
