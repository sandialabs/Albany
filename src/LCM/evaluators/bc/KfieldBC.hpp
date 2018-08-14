//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef KFIELDBC_HPP
#define KFIELDBC_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Teuchos_ParameterList.hpp"

#include <vector>
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace LCM {
/** \brief KfieldBC Dirichlet evaluator

*/

// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template <typename EvalT, typename Traits>
class KfieldBC;

template <typename EvalT, typename Traits>
class KfieldBC_Base : public PHAL::DirichletBase<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;
  KfieldBC_Base(Teuchos::ParameterList& p);
  ScalarT&
  getValue(const std::string& n);
  void
  computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval, RealType time);

  RealType    mu, nu, KIval, KIIval;
  ScalarT     KI, KII;
  std::string KI_name, KII_name;

 protected:
  const int             offset;
  std::vector<RealType> timeValues;
  std::vector<RealType> KIValues;
  std::vector<RealType> KIIValues;
};

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Residual, Traits>
    : public KfieldBC_Base<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template <typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public KfieldBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template <typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Tangent, Traits>
    : public KfieldBC_Base<PHAL::AlbanyTraits::Tangent, Traits>
{
 public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template <typename Traits>
class KfieldBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public KfieldBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
 public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif
