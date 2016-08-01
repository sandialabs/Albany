//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Core_Integrator_hpp)
#define Core_Integrator_hpp

namespace CP
{

template<typename EvalT, Intrepid2::Index NumDimT, Intrepid2::Index NumSlipT>
class Integrator
{
  public:

    using ScalarT = typename EvalT::ScalarT;
    
    Integrator(
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt)
      : slip_systems_(slip_systems),
        slip_families_(slip_families),
        plasticity_state_(plasticity_state),
        slip_state_(slip_state),
        C_(C),
        F_np1_(F_np1),
        dt_(dt)
    {}

    virtual void update(Intrepid2::Vector<ScalarT, NumSlipT> & residual) const = 0;

  protected:
 
    std::vector<SlipSystem<NumDimT>> const & slip_systems_;
    std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families_;
    PlasticityState<ScalarT, NumDimT> & plasticity_state_;
    SlipState<ScalarT, NumSlipT> & slip_state_;
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
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);

    virtual void update(Intrepid2::Vector<ScalarT, NumSlipT> & residual) const override;

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

    ImplicitIntegrator(
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);
    
    void reevaluateState();

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
class ImplicitSlipIntegrator : public ImplicitIntegrator<EvalT, NumDimT, NumSlipT>
{
  public:
    
    using Base = Integrator<EvalT, NumDimT, NumSlipT>;
    using ScalarT = typename Base::ScalarT;

    ImplicitSlipIntegrator(
      std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
      std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
      PlasticityState<ScalarT, NumDimT> & plasticity_state,
      SlipState<ScalarT, NumSlipT > & slip_state,
      Intrepid2::Tensor4<ScalarT, NumDimT> const & C,
      Intrepid2::Tensor<ScalarT, NumDimT> const & F_np1,
      RealType dt);

    virtual void update(Intrepid2::Vector<ScalarT, NumSlipT> & residual) const override;

  protected:

    using Base::slip_systems_;
    using Base::slip_families_;
    using Base::plasticity_state_;
    using Base::slip_state_;
    using Base::C_;
    using Base::F_np1_;
    using Base::dt_;
};
}

#include "Integrator_Def.hpp"

#endif

