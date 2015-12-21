//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_CrystalPlasticityModel_hpp)
#define LCM_CrystalPlasticityModel_hpp

#include "CrystalPlasticityCore.hpp"
#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "LCM/models/ConstitutiveModel.hpp"
#include <Intrepid_MiniTensor.h>
#include "Intrepid_MiniTensor_Solvers.h"
#include <MiniNonlinearSolver.h>

namespace LCM
{

//! \brief CrystalPlasticity Plasticity Constitutive Model
template<typename EvalT, typename Traits>
class CrystalPlasticityModel: public LCM::ConstitutiveModel<EvalT, Traits>
{
public:

  enum IntegrationScheme
  {
    EXPLICIT = 0, IMPLICIT = 1
  };

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // typedef for automatic differentiation type used in internal Newton loop
  // options are:  DFad (dynamically sized), SFad (static size), SLFad (bounded)
  // typedef typename Sacado::Fad::DFad<ScalarT> Fad;
  // typedef typename Sacado::Fad::SFad<ScalarT, CP::MAX_NUM_SLIP> Fad;
  typedef typename Sacado::Fad::SLFad<ScalarT, CP::MAX_NUM_SLIP> Fad;

  using ConstitutiveModel<EvalT, Traits>::num_dims_;
  using ConstitutiveModel<EvalT, Traits>::num_pts_;
  using ConstitutiveModel<EvalT, Traits>::field_name_map_;

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
      RealType c11_, c12_, c44_, ctest_;
      Intrepid::Tensor4<RealType, CP::MAX_NUM_DIM> C_;
      Intrepid::Tensor<RealType, CP::MAX_NUM_DIM> orientation_;

      ///
      /// Number of slip systems
      ///
      int num_slip_;

      ///
      /// Crystal Plasticity parameters
      ///
      RealType rateSlipReference_, exponentRate_, energyActivation_, H_, Rd_, tau_critical_,
        resistanceSlipInitial_, rateHardening_, stressSaturationInitial_,
        exponentSaturation_;
      std::vector< CP::SlipSystemStruct<CP::MAX_NUM_DIM,CP::MAX_NUM_SLIP> > 
      slip_systems_;

      IntegrationScheme integration_scheme_;
      FlowRule flowRule_;
      HardeningLaw hardeningLaw_;
      RealType implicit_nonlinear_solver_relative_tolerance_;
      RealType implicit_nonlinear_solver_absolute_tolerance_;
      int implicit_nonlinear_solver_max_iterations_;
      bool apply_slip_predictor_;
      int verbosity_;
      bool write_data_file_;
    };
  }

#endif
