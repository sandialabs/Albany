//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_EquilibriumConcentrationBC_hpp)
#define LCM_EquilibriumConcentrationBC_hpp

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_config.hpp>

#include "Teuchos_ParameterList.hpp"

#include <vector>
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"

namespace LCM {
/** \brief Equilibrium Concentration BC Dirichlet evaluator
 */

//------------------------------------------------------------------------------
// Specialization of the DirichletBase class
//
template <typename EvalT, typename Traits>
class EquilibriumConcentrationBC;

template <typename EvalT, typename Traits>
class EquilibriumConcentrationBC_Base
    : public PHAL::DirichletBase<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;
  EquilibriumConcentrationBC_Base(Teuchos::ParameterList& p);
  void
  computeBCs(ScalarT& pressure, ScalarT& Cval);

  RealType applied_conc_, pressure_fac_;

 protected:
  const int coffset_;
  const int poffset_;
};

//------------------------------------------------------------------------------
// Residual
//
template <typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::Residual, Traits>
    : public EquilibriumConcentrationBC_Base<
          PHAL::AlbanyTraits::Residual,
          Traits>
{
 public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Jacobian
//
template <typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public EquilibriumConcentrationBC_Base<
          PHAL::AlbanyTraits::Jacobian,
          Traits>
{
 public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Tangent
//
template <typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::Tangent, Traits>
    : public EquilibriumConcentrationBC_Base<
          PHAL::AlbanyTraits::Tangent,
          Traits>
{
 public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

//------------------------------------------------------------------------------
// Distributed Parameter Derivative
//
template <typename Traits>
class EquilibriumConcentrationBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public EquilibriumConcentrationBC_Base<
          PHAL::AlbanyTraits::DistParamDeriv,
          Traits>
{
 public:
  EquilibriumConcentrationBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif
