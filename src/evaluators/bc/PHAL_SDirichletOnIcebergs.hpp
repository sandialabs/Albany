//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(PHAL_SDirichletOnIcebergs_hpp)
#define PHAL_SDirichletOnIcebergs_hpp

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {

///
/// Strong Dirichlet boundary condition evaluator
///
template<typename EvalT, typename Traits>
class SDirichletOnIcebergs {
  protected:
    std::vector<int> nodes; 
};

//
// Specializations for different Albany Traits.
//

//
// Residual
//
template<typename Traits>
class SDirichletOnIcebergs<PHAL::AlbanyTraits::Residual, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits> {
 public:
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;

  SDirichletOnIcebergs(Teuchos::ParameterList& p);

  void
  preEvaluate(typename Traits::EvalData d);

  void
  evaluateFields(typename Traits::EvalData d);

  protected:
    Teuchos::RCP<Teuchos::Time> timer_gatherAll;
    Teuchos::RCP<Teuchos::Time> timer_zoltan2Icebergs;
};

//
// Jacobian
//
template<typename Traits>
class SDirichletOnIcebergs<PHAL::AlbanyTraits::Jacobian, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits> {
 public:
  using ScalarT = typename PHAL::AlbanyTraits::Jacobian::ScalarT;

  SDirichletOnIcebergs(Teuchos::ParameterList& p);
  
  void
  evaluateFields(typename Traits::EvalData d);

  void 
  set_row_and_col_is_dbc(typename Traits::EvalData d); 

 protected:
  double scale;
  Teuchos::RCP<Tpetra::Vector<int, Tpetra_LO, Tpetra_GO, KokkosNode>> row_is_dbc_; 
  Teuchos::RCP<Tpetra::Vector<int, Tpetra_LO, Tpetra_GO, KokkosNode>> col_is_dbc_; 
  std::vector<int> nodes; 
};

//
// Tangent
//
template<typename Traits>
class SDirichletOnIcebergs<PHAL::AlbanyTraits::Tangent, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits> {
 public:
  using ScalarT = typename PHAL::AlbanyTraits::Tangent::ScalarT;

  SDirichletOnIcebergs(Teuchos::ParameterList& p);

  void
  evaluateFields(typename Traits::EvalData d);

 protected:
  double scale;
  std::vector<int> nodes; 
};

//
// Distributed Parameter Derivative
//
template<typename Traits>
class SDirichletOnIcebergs<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
 public:
  using ScalarT = typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT;

  SDirichletOnIcebergs(Teuchos::ParameterList& p);

  void
  evaluateFields(typename Traits::EvalData d);
 
 protected:
  std::vector<int> nodes; 
};

}  // namespace PHAL

#endif  // PHAL_SDirichletOnIcebergs_hpp
