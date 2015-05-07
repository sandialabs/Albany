//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzBC_hpp)
#define LCM_SchwarzBC_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Vector.h"
#endif

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"

namespace LCM {

//
// \brief Schwarz for models BC Dirichlet evaluator
//

//
// Specialization of the DirichletBase class
//
template<typename EvalT, typename Traits> class SchwarzBC;

template <typename EvalT, typename Traits>
class SchwarzBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;

  SchwarzBC_Base(Teuchos::ParameterList & p);

  void
  setDiscretization(Teuchos::RCP<Albany::AbstractDiscretization> & d)
  {
    disc_ = d;
  }

  Teuchos::RCP<Albany::AbstractDiscretization>
  getDiscretization() const
  {
    return disc_;
  }

  void
  setCoupledAppName(std::string const & can)
  {
    coupled_app_name_ = can;
  }

  std::string
  getCoupledAppName() const
  {
    return coupled_app_name_;
  }

  int
  appIndexFromName(std::string const & name) const
  {
    return std::atoi(name.c_str());
  }

  void
  setCoupledBlockName(std::string const & cbn)
  {
    coupled_block_name_ = cbn;
  }

  std::string
  getCoupledBlockName() const
  {
    return coupled_block_name_;
  }

protected:

  Teuchos::RCP<Albany::Application>
  app_;

  std::string
  coupled_app_name_;

  std::string
  coupled_block_name_;

  Teuchos::RCP<Albany::AbstractDiscretization>
  disc_;
};

//
// Residual
//
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::Residual,Traits>
  : public SchwarzBC_Base<PHAL::AlbanyTraits::Residual, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Jacobian
//
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::Jacobian,Traits>
   : public SchwarzBC_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Tangent
//
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public SchwarzBC_Base<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Distributed Parameter Derivative
//
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public SchwarzBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Stochastic Galerkin Residual
//
#ifdef ALBANY_SG_MP
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::SGResidual,Traits>
   : public SchwarzBC_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Stochastic Galerkin Jacobian
//
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public SchwarzBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Stochastic Galerkin Tangent
//
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::SGTangent,Traits>
   : public SchwarzBC_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Multi-point Residual
//
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::MPResidual,Traits>
   : public SchwarzBC_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Multi-point Jacobian
//
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public SchwarzBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Multi-point Tangent
//
template<typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::MPTangent,Traits>
   : public SchwarzBC_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  SchwarzBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

#endif //ALBANY_SG_MP

}

#endif // LCM_SchwarzBC_hpp
