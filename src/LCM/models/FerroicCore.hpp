#if !defined(FerroicCore_hpp)
#define FerroicCore_hpp

#include <Intrepid2_MiniTensor.h>
#include <MiniNonlinearSolver.h>

namespace FM
{

static constexpr Intrepid2::Index THREE_D = 3;
static constexpr Intrepid2::Index MAX_VRNT = 27;
static constexpr Intrepid2::Index MAX_TRNS = MAX_VRNT*MAX_VRNT;


enum class IntegrationType
{
  UNDEFINED = 0, EXPLICIT = 1, IMPLICIT = 2
};

enum class ExplicitMethod
{
  UNDEFINED = 0, SCALED_DESCENT = 1, DESCENT_NORM = 2
};


/******************************************************************************/
// Data structures 
/******************************************************************************/

/******************************************************************************/
struct CrystalPhase
/******************************************************************************/
{
  CrystalPhase(Intrepid2::Tensor <RealType, THREE_D>& R, 
               Intrepid2::Tensor4<RealType, THREE_D>& C, 
               Intrepid2::Tensor3<RealType, THREE_D>& h, 
               Intrepid2::Tensor <RealType, THREE_D>& eps);

  Intrepid2::Tensor4<RealType, THREE_D> C;
  Intrepid2::Tensor3<RealType, THREE_D> h;
  Intrepid2::Tensor <RealType, THREE_D> b;
  Intrepid2::Tensor <RealType, THREE_D> basis;
};


/******************************************************************************/
struct CrystalVariant
/******************************************************************************/
{
  Intrepid2::Tensor4<RealType, THREE_D> C;
  Intrepid2::Tensor3<RealType, THREE_D> h;
  Intrepid2::Tensor<RealType, THREE_D> b;
  Intrepid2::Tensor<RealType, THREE_D> R;
  Intrepid2::Tensor<RealType, THREE_D> spontStrain;
  Intrepid2::Vector<RealType, THREE_D> spontEDisp;
};

/******************************************************************************/
struct Transition
/******************************************************************************/
{
  Transition() {}
  
//  Teuchos::RCP<CrystalVariant> fromVariant;
//  Teuchos::RCP<CrystalVariant> toVariant;

  Intrepid2::Tensor<RealType, THREE_D> transStrain;
  Intrepid2::Vector<RealType, THREE_D> transEDisp;
};




/******************************************************************************/
// Service functions:
/******************************************************************************/

template<typename DataT>
void 
changeBasis(      Intrepid2::Tensor4<DataT, THREE_D>& inMatlBasis,
            const Intrepid2::Tensor4<DataT, THREE_D>& inGlblBasis,
            const Intrepid2::Tensor <DataT, THREE_D>& Basis);

template<typename DataT>
void 
changeBasis(      Intrepid2::Tensor3<DataT, THREE_D>& inMatlBasis,
            const Intrepid2::Tensor3<DataT, THREE_D>& inGlblBasis,
            const Intrepid2::Tensor <DataT, THREE_D>& Basis);

template<typename DataT>
void 
changeBasis(      Intrepid2::Tensor <DataT, THREE_D>& inMatlBasis,
            const Intrepid2::Tensor <DataT, THREE_D>& inGlblBasis,
            const Intrepid2::Tensor <DataT, THREE_D>& Basis);

template<typename DataT>
void 
changeBasis(      Intrepid2::Vector <DataT, THREE_D>& inMatlBasis,
            const Intrepid2::Vector <DataT, THREE_D>& inGlblBasis,
            const Intrepid2::Tensor <DataT, THREE_D>& Basis);


template<typename NLS, typename DataT>
void 
DescentNorm(NLS & nls, Intrepid2::Vector<DataT, MAX_TRNS> & xi);

template<typename NLS, typename DataT>
void 
ScaledDescent(NLS & nls, Intrepid2::Vector<DataT, MAX_TRNS> & xi);

template<typename DataT, typename ArgT>
void
computeBinFractions(
    Intrepid2::Vector<ArgT, FM::MAX_TRNS> const & xi,
    Teuchos::Array<ArgT>                        & newFractions,
    Teuchos::Array<DataT>                 const & oldFractions,
    Teuchos::Array<int>                   const & transitionMap,
    Kokkos::DynRankView<DataT>            const & aMatrix);


template<typename ArgT>
void
computeInitialState(
    Teuchos::Array<RealType>            const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants,
    Intrepid2::Tensor<ArgT,FM::THREE_D> const & x, 
    Intrepid2::Tensor<ArgT,FM::THREE_D>       & X, 
    Intrepid2::Tensor<ArgT,FM::THREE_D>       & linear_x,
    Intrepid2::Vector<ArgT,FM::THREE_D> const & E, 
    Intrepid2::Vector<ArgT,FM::THREE_D>       & D, 
    Intrepid2::Vector<ArgT,FM::THREE_D>       & linear_D);


template<typename ArgT>
void
computeRelaxedState(
    Teuchos::Array<ArgT>                const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants,
    Intrepid2::Tensor<ArgT,FM::THREE_D> const & x, 
    Intrepid2::Tensor<ArgT,FM::THREE_D>       & X, 
    Intrepid2::Tensor<ArgT,FM::THREE_D>       & linear_x,
    Intrepid2::Vector<ArgT,FM::THREE_D>       & E, 
    Intrepid2::Vector<ArgT,FM::THREE_D> const & D, 
    Intrepid2::Vector<ArgT,FM::THREE_D>       & linear_D);


template<typename DataT, typename ArgT>
void
computeResidual(
    Intrepid2::Vector<ArgT, FM::MAX_TRNS>       & residual,
    Teuchos::Array<ArgT>                  const & fractions,
    Teuchos::Array<int>                   const & transitionMap,
    Teuchos::Array<FM::Transition>        const & transitions,
    Teuchos::Array<FM::CrystalVariant>    const & crystalVariants,
    Teuchos::Array<DataT>                 const & tBarrier,
    Kokkos::DynRankView<DataT>            const & aMatrix,
    Intrepid2::Tensor<ArgT,FM::THREE_D>   const & X, 
    Intrepid2::Tensor<ArgT,FM::THREE_D>   const & linear_x,
    Intrepid2::Vector<ArgT,FM::THREE_D>   const & E,
    Intrepid2::Vector<ArgT,FM::THREE_D>   const & linear_D);




/******************************************************************************/
// Host-independent Models
/******************************************************************************/


//
//! Nonlinear Solver (NLS) class for the domain switching / phase transition model.
//  Unknowns: transition rates
//
template<typename EvalT>
class DomainSwitching:
    public Intrepid2::Function_Base<
    DomainSwitching<EvalT>, typename EvalT::ScalarT>
{
  using ArgT = typename EvalT::ScalarT;

public:

  //! Constructor.
  DomainSwitching(
      Teuchos::Array<FM::CrystalVariant> const & crystalVariants,
      Teuchos::Array<FM::Transition>     const & transitions,
      Teuchos::Array<RealType>           const & transBarriers,
      Teuchos::Array<RealType>           const & binFractions,
      Kokkos::DynRankView<RealType>      const & aMatrix,
      Intrepid2::Tensor<ArgT,THREE_D>    const & x,
      Intrepid2::Vector<ArgT,THREE_D>    const & E,
      RealType dt);

  static constexpr char const * const NAME =
      "Domain Switching Nonlinear System";

  //! Default implementation of value function.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  T
  value(Intrepid2::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector as a function of the 
  // transition rate at step N+1.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const;


  //! Default implementation of hessian function.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x);

  int getNumStates(){return m_numActiveTransitions;}

private:

  Teuchos::Array<FM::CrystalVariant>  const & m_crystalVariants;
  Teuchos::Array<FM::Transition>      const & m_transitions;
  Teuchos::Array<RealType>            const & m_transBarriers;
  Teuchos::Array<RealType>            const & m_binFractions;
  Kokkos::DynRankView<RealType>       const & m_aMatrix;
  Intrepid2::Tensor<ArgT,THREE_D>     const & m_x;
  Intrepid2::Vector<ArgT,THREE_D>             m_D;
  RealType m_dt;
  int m_numActiveTransitions;

  Teuchos::Array<int> m_transitionMap;
};

} // namespace FM

#include "FerroicCore_Def.hpp"

#endif
