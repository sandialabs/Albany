//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Core_Integrator_hpp)
#define Core_Integrator_hpp

#include "CrystalPlasticityFwd.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.h"
#include <ROL_MiniTensor_MiniSolver.hpp>

namespace CP
{

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
class Integrator
{
  public:

    using ScalarT = typename EvalT::ScalarT;
    Integrator(
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity)
      : nox_status_test_(nox_status_test),
        num_slip_(state_internal.slip_n_.get_dimension()),
        num_dims_(state_mechanical.Fp_n_.get_dimension()),
        num_iters_(0),
        norm_residual_(0.0),
        slip_systems_(slip_systems),
        slip_families_(slip_families),
        state_mechanical_(state_mechanical),
        state_internal_(state_internal),
        C_(C),
        dt_(dt),
        verbosity_(verbosity)
    {}

    virtual ~Integrator(){}

    virtual void update() const = 0;

    void forceGlobalLoadStepReduction(
      std::string const & message) const;

    void setWarningStatus(
      std::string const & message) const;
  
    int
    getNumIters() const { return num_iters_; }

    RealType
    getNormResidual() const { return norm_residual_; }

    NOX::StatusTest::StatusType
    getStatus() const { return nox_status_test_->status_; }

    std::string
    getMessage() const { return nox_status_test_->status_message_; }

  protected:

    Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag>
    nox_status_test_;

    int
    num_slip_;

    int
    num_dims_;

    mutable int
    num_iters_;

    mutable RealType
    norm_residual_;

    std::vector<SlipSystem<NumDimT>> const &
    slip_systems_;

    std::vector<SlipFamily<NumDimT, NumSlipT>> const &
    slip_families_;

    StateMechanical<ScalarT, NumDimT> &
    state_mechanical_;

    StateInternal<ScalarT, NumSlipT> &
    state_internal_;

    minitensor::Tensor4<ScalarT, NumDimT> const &
    C_;

    RealType
    dt_;

    CP::Verbosity const 
    verbosity_;
};


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
class IntegratorFactory
{
  public:

    using ScalarT = typename EvalT::ScalarT;
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using Minimizer = minitensor::Minimizer<ValueT, CP::NlsDim<NumSlipT>::value>;
    using RolMinimizer = ROL::MiniTensor_Minimizer<ValueT, CP::NlsDim<NumSlipT>::value>;
    using IntegratorBase = Integrator<EvalT, NumDimT, NumSlipT>;

    IntegratorFactory(
      utility::StaticAllocator & allocator,
      const Minimizer & minimizer,
      const RolMinimizer & rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      CP::StateMechanical<ScalarT, NumDimT> & state_mechanical,
      CP::StateInternal<ScalarT, NumSlipT> & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity);

    utility::StaticPointer<IntegratorBase>
    operator()(CP::IntegrationScheme integration_scheme,
               CP::ResidualType residual_type) const;

  private:

    utility::StaticAllocator &
    allocator_;

    const Minimizer &
    minimizer_;

    const RolMinimizer &
    rol_minimizer_;

    minitensor::StepType
    step_type_;

    Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag>
    nox_status_test_;

    std::vector<CP::SlipSystem<NumDimT>> const &
    slip_systems_;

    std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const &
    slip_families_;

    CP::StateMechanical<ScalarT, NumDimT> &
    state_mechanical_;

    CP::StateInternal<ScalarT, NumSlipT> &
    state_internal_;

    minitensor::Tensor4<ScalarT, NumDimT> const &
    C_;

    RealType
    dt_;

    CP::Verbosity const
    verbosity_;
};

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
class ExplicitIntegrator : public Integrator<EvalT, NumDimT, NumSlipT>
{
  public:

    using Base = Integrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;

    ExplicitIntegrator(
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity);

    virtual void
    update() const override;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::state_mechanical_;
    using Base::state_internal_;
    using Base::C_;
    using Base::dt_;
    using Base::verbosity_;
  };

template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
class ImplicitIntegrator : public Integrator<EvalT, NumDimT, NumSlipT>
{
  public:

    using Base = Integrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using Minimizer = minitensor::Minimizer<ValueT, CP::NlsDim<NumSlipT>::value>;
    using RolMinimizer = ROL::MiniTensor_Minimizer<ValueT, CP::NlsDim<NumSlipT>::value>;

    ImplicitIntegrator(
      const Minimizer & minimizer,
      const RolMinimizer & rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity);

    void
    reevaluateState() const;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::state_mechanical_;
    using Base::state_internal_;
    using Base::C_;
    using Base::dt_;
    using Base::verbosity_;

    bool
    using_rol_minimizer_;

    mutable Minimizer
    minimizer_;

    mutable RolMinimizer
    rol_minimizer_;

    minitensor::StepType
    step_type_;
};


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
class ImplicitSlipIntegrator : public ImplicitIntegrator<EvalT, NumDimT, NumSlipT>
{
  public:

    using Base = ImplicitIntegrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;
    using ValueT = typename Base::ValueT;
    using Minimizer = typename Base::Minimizer;
    using RolMinimizer = typename Base::RolMinimizer;

    ImplicitSlipIntegrator(
      const Minimizer &minimizer,
      const RolMinimizer &rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity);

    virtual void
    update() const override;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::state_mechanical_;
    using Base::state_internal_;
    using Base::C_;
    using Base::dt_;
    using Base::minimizer_;
    using Base::rol_minimizer_;
    using Base::step_type_;
    using Base::verbosity_;
};


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
class ImplicitSlipHardnessIntegrator : public ImplicitIntegrator<EvalT, NumDimT, NumSlipT>
{
  public:

    using Base = ImplicitIntegrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;
    using ValueT = typename Base::ValueT;
    using Minimizer = typename Base::Minimizer;
    using RolMinimizer = typename Base::RolMinimizer;

    ImplicitSlipHardnessIntegrator(
      const Minimizer &minimizer,
      const RolMinimizer &rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity);

    virtual void
    update() const override;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::state_mechanical_;
    using Base::state_internal_;
    using Base::C_;
    using Base::dt_;
    using Base::rol_minimizer_;
    using Base::minimizer_;
    using Base::step_type_;
    using Base::verbosity_;
};


template<typename EvalT, minitensor::Index NumDimT, minitensor::Index NumSlipT>
class ImplicitConstrainedSlipHardnessIntegrator : public ImplicitIntegrator<EvalT, NumDimT, NumSlipT>
{
  public:

    using Base = ImplicitIntegrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;
    using ValueT = typename Base::ValueT;
    using Minimizer = typename Base::Minimizer;
    using RolMinimizer = typename Base::RolMinimizer;

    ImplicitConstrainedSlipHardnessIntegrator(
      const Minimizer &minimizer,
      const RolMinimizer &rol_minimizer,
      minitensor::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      StateMechanical<ScalarT, NumDimT> & state_mechanical,
      StateInternal<ScalarT, NumSlipT > & state_internal,
      minitensor::Tensor4<ScalarT, NumDimT> const & C,
      RealType dt,
      CP::Verbosity const verbosity);

    virtual void
    update() const override;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::state_mechanical_;
    using Base::state_internal_;
    using Base::C_;
    using Base::dt_;
    using Base::minimizer_;
    using Base::rol_minimizer_;
    using Base::step_type_;
    using Base::verbosity_;
};
}

#include "Integrator_Def.hpp"

#endif

