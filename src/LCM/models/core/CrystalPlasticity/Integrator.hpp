//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Core_Integrator_hpp)
#define Core_Integrator_hpp

#include "CrystalPlasticityFwd.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.h"

namespace CP
{

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
class Integrator
{
  public:

    using ScalarT = typename EvalT::ScalarT;
    Integrator(
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
      : nox_status_test_(nox_status_test),
        num_slip_(slip_state.slip_n_.get_dimension()),
        num_dims_(plasticity_state.Fp_n_.get_dimension()),
        num_iters_(0),
        slip_systems_(slip_systems),
        slip_families_(slip_families),
        plasticity_state_(plasticity_state),
        slip_state_(slip_state),
        C_(C),
        F_np1_(F_np1),
        dt_(dt)
    {}

    virtual bool update(RealType & residual_norm) const = 0;

    void forceGlobalLoadStepReduction() const;
    int getNumIters() const { return num_iters_; }

  protected:

    Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test_;
 
    int num_slip_;
    int num_dims_;
    mutable int num_iters_;

    std::vector<SlipSystem<NumDimT>> const & slip_systems_;
    std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families_;
    PlasticityState<ScalarT, NumDimT> & plasticity_state_;
    SlipState<ScalarT, NumSlipT> & slip_state_;
    Intrepid2::Tensor4<ScalarT, NumDimT> const & C_;
    Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1_;
    RealType dt_;
};


template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
class IntegratorFactory
{
  public:
    
    using ScalarT = typename EvalT::ScalarT;
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using Minimizer = Intrepid2::Minimizer<ValueT, CP::NlsDim<NumSlipT>::value>;
    using IntegratorBase = Integrator<EvalT, NumDimT, NumSlipT>;

    IntegratorFactory(utility::StaticAllocator & allocator,
      const Minimizer & minimizer,
      Intrepid2::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      CP::PlasticityState<ScalarT, NumDimT> & plasticity_state,
      CP::SlipState<ScalarT, NumSlipT> & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);

    utility::StaticPointer<IntegratorBase>
    operator()(CP::IntegrationScheme integration_scheme,
               CP::ResidualType residual_type) const;

  private:

    utility::StaticAllocator & allocator_;

    const Minimizer & minimizer_;
    Intrepid2::StepType step_type_;
    Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test_;

    std::vector<CP::SlipSystem<NumDimT>> const & slip_systems_;
    std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families_;

    CP::PlasticityState<ScalarT, NumDimT> & plasticity_state_;
    CP::SlipState<ScalarT, NumSlipT> & slip_state_;
    
    Intrepid2::Tensor4<ScalarT, NumDimT> const & C_;
    Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1_;
    RealType dt_;
};

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
class ExplicitIntegrator : public Integrator<EvalT, NumDimT, NumSlipT>
{
  public:
    
    using Base = Integrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;

    ExplicitIntegrator(
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);

    virtual bool update(RealType & residual_norm) const override;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::plasticity_state_;
    using Base::slip_state_;
    using Base::C_;
    using Base::F_np1_;
    using Base::dt_;
};

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
class ImplicitIntegrator : public Integrator<EvalT, NumDimT, NumSlipT>
{
  public:
    
    using Base = Integrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using Minimizer = Intrepid2::Minimizer<ValueT, CP::NlsDim<NumSlipT>::value>;

    ImplicitIntegrator(
      const Minimizer & minimizer,
      Intrepid2::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);
    
    bool reevaluateState(RealType & residual_norm) const;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::plasticity_state_;
    using Base::slip_state_;
    using Base::C_;
    using Base::F_np1_;
    using Base::dt_;

    mutable Minimizer minimizer_;
    Intrepid2::StepType step_type_;
};


template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
class ImplicitSlipIntegrator : public ImplicitIntegrator<EvalT, NumDimT, NumSlipT>
{
  public:
    
    using Base = ImplicitIntegrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;
    using ValueT = typename Base::ValueT;
    using Minimizer = typename Base::Minimizer;

    ImplicitSlipIntegrator(
      const Minimizer &minimizer,
      Intrepid2::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);

    virtual bool update(RealType & residual_norm) const override;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::plasticity_state_;
    using Base::slip_state_;
    using Base::C_;
    using Base::F_np1_;
    using Base::dt_;
    using Base::minimizer_;
    using Base::step_type_;
};


template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
class ImplicitSlipHardnessIntegrator : public ImplicitIntegrator<EvalT, NumDimT, NumSlipT>
{
  public:
    
    using Base = ImplicitIntegrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;
    using ValueT = typename Base::ValueT;
    using Minimizer = typename Base::Minimizer;

    ImplicitSlipHardnessIntegrator(
      const Minimizer &minimizer,
      Intrepid2::StepType step_type,
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> nox_status_test,
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);

    virtual bool update(RealType & residual_norm) const override;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::plasticity_state_;
    using Base::slip_state_;
    using Base::C_;
    using Base::F_np1_;
    using Base::dt_;
    using Base::minimizer_;
    using Base::step_type_;
};
}

#include "Integrator_Def.hpp"

#endif

