//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzBC_hpp)
#define LCM_SchwarzBC_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"

namespace LCM {

//
// \brief Torsion BC Dirichlet evaluator
//

//
// Specialization of the DirichletBase class
//
template<typename EvalT, typename Traits> class SchwarzBC;

template <typename EvalT, typename Traits>
class SchwarzBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef Teuchos::RCP<Albany::AbstractDiscretization> Discretization;

  SchwarzBC_Base(Teuchos::ParameterList & p);

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
  setCoupledBlock(std::string const & cb) {coupled_block_ = cb;}

  std::string
  getCoupledBlock() const {return coupled_block_;}

protected:

  std::string
  coupled_block_;

  Discretization
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
