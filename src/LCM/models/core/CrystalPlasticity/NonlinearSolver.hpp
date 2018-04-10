//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Core_NonlinearSolver_hpp)
#define Core_NonlinearSolver_hpp

#include <MiniNonlinearSolver.h>
#include "CrystalPlasticityCore.hpp"

namespace CP
{
  /**
   *  Nonlinear Solver (NLS) class for the CrystalPlasticity model; slip
   *  increments as unknowns.
   */
  template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
  class ResidualSlipNLS:
      public minitensor::Function_Base<
      ResidualSlipNLS<NumDimT, NumSlipT, EvalT>,
      typename EvalT::ScalarT, NumSlipT>
  {
    using ScalarT = typename EvalT::ScalarT;

  public:

    //! Constructor.
    ResidualSlipNLS(
        minitensor::Tensor4<ScalarT, NumDimT> const & C,
        std::vector<SlipSystem<NumDimT>> const & slip_systems,
        std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
        minitensor::Tensor<RealType, NumDimT> const & Fp_n,
        minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
        minitensor::Vector<RealType, NumSlipT> const & slip_n,
        minitensor::Tensor<ScalarT, NumDimT> const & F_np1,
        RealType dt,
        Verbosity verbosity);

    static constexpr char const * const
    NAME{"Crystal Plasticity Nonlinear System"};

    using Base = minitensor::Function_Base<
        ResidualSlipNLS<NumDimT, NumSlipT, EvalT>,
        typename EvalT::ScalarT, NumSlipT>;

    //! Default implementation of value.
    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    T
    value(minitensor::Vector<T, N> const & x);

    //! Gradient function; returns the residual vector as a function of the slip
    // at step N+1.
    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    minitensor::Vector<T, N>
    gradient(minitensor::Vector<T, N> const & x);


    //! Default implementation of hessian.
    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    minitensor::Tensor<T, N>
    hessian(minitensor::Vector<T, N> const & x);

  private:

    RealType
    num_dim_;

    RealType
    num_slip_;

    minitensor::Tensor4<ScalarT, NumDimT> const &
    C_;

    std::vector<SlipSystem<NumDimT>> const &
    slip_systems_;

    std::vector<SlipFamily<NumDimT, NumSlipT>> const &
    slip_families_;

    minitensor::Tensor<RealType, NumDimT> const &
    Fp_n_;

    minitensor::Vector<RealType, NumSlipT> const &
    state_hardening_n_;

    minitensor::Vector<RealType, NumSlipT> const &
    slip_n_;

    minitensor::Tensor<ScalarT, NumDimT> const &
    F_np1_;

    RealType
    dt_;

    Verbosity
    verbosity_;
  };

  //
  //  Dissipation class for the CrystalPlasticity model; slip
  //  increments as unknowns.
  //
  template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
  class Dissipation:
      public minitensor::Function_Base<
      Dissipation<NumDimT, NumSlipT, EvalT>,
      typename EvalT::ScalarT, NumSlipT>
  {
      using ScalarT = typename EvalT::ScalarT;

  public:

    //! Constructor.
    Dissipation(
        std::vector<CP::SlipSystem<NumDimT>> const & slip_systems,
        std::vector<CP::SlipFamily<NumDimT, NumSlipT>> const & slip_families,
        minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
        minitensor::Vector<RealType, NumSlipT> const & slip_n,
        minitensor::Tensor<RealType, NumDimT> const & F_n,
        minitensor::Tensor<ScalarT, NumDimT> const & F_np1,
        RealType dt,
        Verbosity verbosity);

    static constexpr char const * const
    NAME{"Dissipation with slip"};

    using Base = minitensor::Function_Base<
        Dissipation<NumDimT, NumSlipT, EvalT>,
        typename EvalT::ScalarT, NumSlipT>;



    //! Default implementation of value.
    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    T
    value(minitensor::Vector<T, N> const & x);

    //! Gradient function; returns the residual vector as a function of the slip
    // at step N+1.
    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    minitensor::Vector<T, N>
    gradient(minitensor::Vector<T, N> const & x);


    //! Default implementation of hessian.
    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    minitensor::Tensor<T, N>
    hessian(minitensor::Vector<T, N> const & x);

  private:

    RealType
    num_dim_;

    RealType
    num_slip_;

    std::vector<SlipSystem<NumDimT>> const &
    slip_systems_;

    std::vector<SlipFamily<NumDimT, NumSlipT>> const &
    slip_families_;

    minitensor::Vector<RealType, NumSlipT> const &
    state_hardening_n_;

    minitensor::Vector<RealType, NumSlipT> const &
    slip_n_;

    minitensor::Tensor<RealType, NumDimT> const &
    F_n_;

    minitensor::Tensor<ScalarT, NumDimT> const &
    F_np1_;

    RealType
    dt_;

    Verbosity
    verbosity_;
  };

  //
  //! Nonlinear Solver (NLS) class for the CrystalPlasticity model; slip
  //  increments and hardnesses as unknowns.
  //
  template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
  class ResidualSlipHardnessNLS:
      public minitensor::Function_Base<
      ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>,
      typename EvalT::ScalarT, CP::NlsDim<NumSlipT>::value>
  {
    using ScalarT = typename EvalT::ScalarT;

  public:

    //! Constructor.
    ResidualSlipHardnessNLS(
        minitensor::Tensor4<ScalarT, NumDimT> const & C,
        std::vector<SlipSystem<NumDimT>> const & slip_systems,
        std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
        minitensor::Tensor<RealType, NumDimT> const & Fp_n,
        minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
        minitensor::Vector<RealType, NumSlipT> const & slip_n,
        minitensor::Tensor<ScalarT, NumDimT> const & F_np1,
        RealType dt,
        Verbosity verbosity);

    static constexpr char const * const
    NAME{"Slip and Hardness Residual Nonlinear System"};

    using Base = minitensor::Function_Base<
        ResidualSlipHardnessNLS<NumDimT, NumSlipT, EvalT>,
        typename EvalT::ScalarT, CP::NlsDim<NumSlipT>::value>;

    //! Default implementation of value.
    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    T
    value(minitensor::Vector<T, N> const & x);

    //! Gradient function; returns the residual vector as a function of the slip
    // and hardness at step N+1.
    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    minitensor::Vector<T, N>
    gradient(minitensor::Vector<T, N> const & x);


    //! Default implementation of hessian.
    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    minitensor::Tensor<T, N>
    hessian(minitensor::Vector<T, N> const & x);

  private:

    RealType
    num_dim_;

    RealType
    num_slip_;

    minitensor::Tensor4<ScalarT, NumDimT> const &
    C_;

    std::vector<SlipSystem<NumDimT>> const &
    slip_systems_;

    std::vector<SlipFamily<NumDimT, NumSlipT>> const &
    slip_families_;

    minitensor::Tensor<RealType, NumDimT> const &
    Fp_n_;

    minitensor::Vector<RealType, NumSlipT> const &
    state_hardening_n_;

    minitensor::Vector<RealType, NumSlipT> const &
    slip_n_;

    minitensor::Tensor<ScalarT, NumDimT> const &
    F_np1_;

    RealType
    dt_;

    Verbosity
    verbosity_;
  };

  template<minitensor::Index NumDimT, minitensor::Index NumSlipT, typename EvalT>
  class ResidualSlipHardnessFN:
    public minitensor::Function_Base<
      ResidualSlipHardnessFN<NumDimT, NumSlipT, EvalT>,
      typename EvalT::ScalarT, CP::NlsDim<NumSlipT>::value>
  {
  public:

    using Base = minitensor::Function_Base<
        ResidualSlipHardnessFN<NumDimT, NumSlipT, EvalT>,
         typename EvalT::ScalarT, CP::NlsDim<NumSlipT>::value>;

    using ScalarT = typename EvalT::ScalarT;

    ResidualSlipHardnessFN(
        minitensor::Tensor4<ScalarT, NumDimT> const & C,
        std::vector<SlipSystem<NumDimT>> const & slip_systems,
        std::vector<SlipFamily<NumDimT, NumSlipT>> const & slip_families,
        minitensor::Tensor<RealType, NumDimT> const & Fp_n,
        minitensor::Vector<RealType, NumSlipT> const & state_hardening_n,
        minitensor::Vector<RealType, NumSlipT> const & slip_n,
        minitensor::Tensor<ScalarT, NumDimT> const & F_np1,
        RealType dt,
        Verbosity verbosity);

    static constexpr char const * const
    NAME{"Crystal Plasticity Function"};

    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    T
    value(minitensor::Vector<T, N> const & x);

    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    minitensor::Vector<T, N>
    gradient(minitensor::Vector<T, N> const & x) {
      return Base::gradient(*this, x);
    }

    template<typename T, minitensor::Index N = minitensor::DYNAMIC>
    minitensor::Tensor<T, N>
    hessian(minitensor::Vector<T, N> const & x) {
      return Base::hessian(*this, x);
    }

  private:

    RealType
    num_dim_;

    RealType
    num_slip_;

    minitensor::Tensor4<ScalarT, NumDimT> const &
    C_;

    std::vector<SlipSystem<NumDimT>> const &
    slip_systems_;

    std::vector<SlipFamily<NumDimT, NumSlipT>> const &
    slip_families_;

    minitensor::Tensor<RealType, NumDimT> const &
    Fp_n_;

    minitensor::Vector<RealType, NumSlipT> const &
    state_hardening_n_;

    minitensor::Vector<RealType, NumSlipT> const &
    slip_n_;

    minitensor::Tensor<ScalarT, NumDimT> const &
    F_np1_;

    RealType
    dt_;

    Verbosity
    verbosity_;
  };

}

#include "NonlinearSolver_Def.hpp"

#endif

