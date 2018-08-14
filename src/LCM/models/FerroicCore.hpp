#if !defined(FerroicCore_hpp)
#define FerroicCore_hpp

#include <MiniNonlinearSolver.h>
#include <MiniTensor.h>

namespace FM {

static constexpr minitensor::Index THREE_D  = 3;
static constexpr minitensor::Index MAX_VRNT = 27;
static constexpr minitensor::Index MAX_TRNS = MAX_VRNT * MAX_VRNT;

enum class IntegrationType
{
  UNDEFINED = 0,
  EXPLICIT  = 1,
  IMPLICIT  = 2
};

enum class ExplicitMethod
{
  UNDEFINED      = 0,
  SCALED_DESCENT = 1,
  DESCENT_NORM   = 2
};

/******************************************************************************/
// Data structures
/******************************************************************************/

/******************************************************************************/
struct CrystalPhase
/******************************************************************************/
{
  CrystalPhase(
      minitensor::Tensor<RealType, THREE_D>&  R,
      minitensor::Tensor4<RealType, THREE_D>& C,
      minitensor::Tensor3<RealType, THREE_D>& h,
      minitensor::Tensor<RealType, THREE_D>&  eps);

  minitensor::Tensor4<RealType, THREE_D> C;
  minitensor::Tensor3<RealType, THREE_D> h;
  minitensor::Tensor<RealType, THREE_D>  b;
  minitensor::Tensor<RealType, THREE_D>  basis;
};

/******************************************************************************/
struct CrystalVariant
/******************************************************************************/
{
  minitensor::Tensor4<RealType, THREE_D> C;
  minitensor::Tensor3<RealType, THREE_D> h;
  minitensor::Tensor<RealType, THREE_D>  b;
  minitensor::Tensor<RealType, THREE_D>  R;
  minitensor::Tensor<RealType, THREE_D>  spontStrain;
  minitensor::Vector<RealType, THREE_D>  spontEDisp;
};

/******************************************************************************/
struct Transition
/******************************************************************************/
{
  Transition() {}

  //  Teuchos::RCP<CrystalVariant> fromVariant;
  //  Teuchos::RCP<CrystalVariant> toVariant;

  minitensor::Tensor<RealType, THREE_D> transStrain;
  minitensor::Vector<RealType, THREE_D> transEDisp;
};

/******************************************************************************/
// Service functions:
/******************************************************************************/

template <typename DataT>
void
changeBasis(
    minitensor::Tensor4<DataT, THREE_D>&       inMatlBasis,
    const minitensor::Tensor4<DataT, THREE_D>& inGlblBasis,
    const minitensor::Tensor<DataT, THREE_D>&  Basis);

template <typename DataT>
void
changeBasis(
    minitensor::Tensor3<DataT, THREE_D>&       inMatlBasis,
    const minitensor::Tensor3<DataT, THREE_D>& inGlblBasis,
    const minitensor::Tensor<DataT, THREE_D>&  Basis);

template <typename DataT>
void
changeBasis(
    minitensor::Tensor<DataT, THREE_D>&       inMatlBasis,
    const minitensor::Tensor<DataT, THREE_D>& inGlblBasis,
    const minitensor::Tensor<DataT, THREE_D>& Basis);

template <typename DataT>
void
changeBasis(
    minitensor::Vector<DataT, THREE_D>&       inMatlBasis,
    const minitensor::Vector<DataT, THREE_D>& inGlblBasis,
    const minitensor::Tensor<DataT, THREE_D>& Basis);

template <typename NLS, typename DataT>
void
DescentNorm(NLS& nls, minitensor::Vector<DataT, MAX_TRNS>& xi);

template <typename NLS, typename DataT>
void
ScaledDescent(NLS& nls, minitensor::Vector<DataT, MAX_TRNS>& xi);

template <typename DataT, typename ArgT>
void
computeBinFractions(
    minitensor::Vector<ArgT, FM::MAX_TRNS> const& xi,
    Teuchos::Array<ArgT>&                         newFractions,
    Teuchos::Array<DataT> const&                  oldFractions,
    Teuchos::Array<int> const&                    transitionMap,
    Kokkos::DynRankView<DataT> const&             aMatrix);

template <typename ArgT>
void
computeInitialState(
    Teuchos::Array<RealType> const&              fractions,
    Teuchos::Array<FM::CrystalVariant> const&    crystalVariants,
    minitensor::Tensor<ArgT, FM::THREE_D> const& x,
    minitensor::Tensor<ArgT, FM::THREE_D>&       X,
    minitensor::Tensor<ArgT, FM::THREE_D>&       linear_x,
    minitensor::Vector<ArgT, FM::THREE_D> const& E,
    minitensor::Vector<ArgT, FM::THREE_D>&       D,
    minitensor::Vector<ArgT, FM::THREE_D>&       linear_D);

template <typename ArgT>
void
computeRelaxedState(
    Teuchos::Array<ArgT> const&                  fractions,
    Teuchos::Array<FM::CrystalVariant> const&    crystalVariants,
    minitensor::Tensor<ArgT, FM::THREE_D> const& x,
    minitensor::Tensor<ArgT, FM::THREE_D>&       X,
    minitensor::Tensor<ArgT, FM::THREE_D>&       linear_x,
    minitensor::Vector<ArgT, FM::THREE_D>&       E,
    minitensor::Vector<ArgT, FM::THREE_D> const& D,
    minitensor::Vector<ArgT, FM::THREE_D>&       linear_D);

template <typename DataT, typename ArgT>
void
computeResidual(
    minitensor::Vector<ArgT, FM::MAX_TRNS>&      residual,
    Teuchos::Array<ArgT> const&                  fractions,
    Teuchos::Array<int> const&                   transitionMap,
    Teuchos::Array<FM::Transition> const&        transitions,
    Teuchos::Array<FM::CrystalVariant> const&    crystalVariants,
    Teuchos::Array<DataT> const&                 tBarrier,
    Kokkos::DynRankView<DataT> const&            aMatrix,
    minitensor::Tensor<ArgT, FM::THREE_D> const& X,
    minitensor::Tensor<ArgT, FM::THREE_D> const& linear_x,
    minitensor::Vector<ArgT, FM::THREE_D> const& E,
    minitensor::Vector<ArgT, FM::THREE_D> const& linear_D);

/******************************************************************************/
// Host-independent Models
/******************************************************************************/

//
//! Nonlinear Solver (NLS) class for the domain switching / phase transition
//! model.
//  Unknowns: transition rates
//
template <typename EvalT, minitensor::Index M = FM::MAX_TRNS>
class DomainSwitching
    : public minitensor::
          Function_Base<DomainSwitching<EvalT, M>, typename EvalT::ScalarT, M>
{
  using ArgT = typename EvalT::ScalarT;

 public:
  //! Constructor.
  DomainSwitching(
      Teuchos::Array<FM::CrystalVariant> const& crystalVariants,
      Teuchos::Array<FM::Transition> const&     transitions,
      Teuchos::Array<RealType> const&           transBarriers,
      Teuchos::Array<RealType> const&           binFractions,
      Kokkos::DynRankView<RealType> const&      aMatrix,
      minitensor::Tensor<ArgT, THREE_D> const&  x,
      minitensor::Vector<ArgT, THREE_D> const&  E,
      RealType                                  dt);

  static constexpr char const* const NAME{"Domain Switching Nonlinear System"};

  using Base = minitensor::
      Function_Base<DomainSwitching<EvalT, M>, typename EvalT::ScalarT, M>;

  //! Default implementation of value function.
  template <typename T, minitensor::Index N = minitensor::DYNAMIC>
  T
  value(minitensor::Vector<T, N> const& x);

  //! Gradient function; returns the residual vector as a function of the
  // transition rate at step N+1.
  template <typename T, minitensor::Index N = minitensor::DYNAMIC>
  minitensor::Vector<T, N>
  gradient(minitensor::Vector<T, N> const& x) const;

  //! Default implementation of hessian function.
  template <typename T, minitensor::Index N = minitensor::DYNAMIC>
  minitensor::Tensor<T, N>
  hessian(minitensor::Vector<T, N> const& x);

  int
  getNumStates()
  {
    return m_numActiveTransitions;
  }

 private:
  Teuchos::Array<FM::CrystalVariant> const& m_crystalVariants;
  Teuchos::Array<FM::Transition> const&     m_transitions;
  Teuchos::Array<RealType> const&           m_transBarriers;
  Teuchos::Array<RealType> const&           m_binFractions;
  Kokkos::DynRankView<RealType> const&      m_aMatrix;
  minitensor::Tensor<ArgT, THREE_D> const&  m_x;
  minitensor::Vector<ArgT, THREE_D>         m_D;
  RealType                                  m_dt;
  int                                       m_numActiveTransitions;

  Teuchos::Array<int> m_transitionMap;
};

}  // namespace FM

#include "FerroicCore_Def.hpp"

#endif
