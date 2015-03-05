//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzModelsBC_hpp)
#define LCM_SchwarzModelsBC_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#ifdef ALBANY_EPETRA
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
template<typename EvalT, typename Traits> class SchwarzModelsBC;

template <typename EvalT, typename Traits>
class SchwarzModelsBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef Teuchos::RCP<Albany::AbstractDiscretization> Discretization;

  SchwarzModelsBC_Base(Teuchos::ParameterList & p);

  void
  computeBCs(
      typename Traits::EvalData dirichlet_workset,
      size_t const ns_node,
      ScalarT & x_val,
      ScalarT & y_val,
      ScalarT & z_val);

  void
  setDiscretization(Discretization & d) {disc_ = d;}

  Discretization
  getDiscretization() const {return disc_;}

  void
  setCoupledModel(std::string const & cm) {coupled_model_ = cm;}

  std::string
  getCoupledModel() const {return coupled_model_;}

protected:

  std::string
  coupled_model_;

  Discretization
  disc_;
};

//
// Residual
//
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::Residual,Traits>
  : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::Residual, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Jacobian
//
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::Jacobian,Traits>
   : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Tangent
//
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Distributed Parameter Derivative
//
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Stochastic Galerkin Residual
//
#ifdef ALBANY_SG_MP
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::SGResidual,Traits>
   : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Stochastic Galerkin Jacobian
//
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Stochastic Galerkin Tangent
//
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::SGTangent,Traits>
   : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Multi-point Residual
//
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::MPResidual,Traits>
   : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Multi-point Jacobian
//
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

//
// Multi-point Tangent
//
template<typename Traits>
class SchwarzModelsBC<PHAL::AlbanyTraits::MPTangent,Traits>
   : public SchwarzModelsBC_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  SchwarzModelsBC(Teuchos::ParameterList & p);
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

#endif //ALBANY_SG_MP

}

#endif // LCM_SchwarzModelsBC_hpp
