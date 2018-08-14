//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzBC_hpp)
#define LCM_SchwarzBC_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "Sacado_ParameterAccessor.hpp"

#if defined(ALBANY_DTK)
#include "DTK_MapOperatorFactory.hpp"
#include "DTK_STKMeshHelpers.hpp"
#include "DTK_STKMeshManager.hpp"
#endif

namespace LCM {

//
// \brief Schwarz for models BC Dirichlet evaluator
//

//
// Specialization of the DirichletBase class
//
template <typename EvalT, typename Traits>
class SchwarzBC;

template <typename EvalT, typename Traits>
class SchwarzBC_Base : public PHAL::DirichletBase<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  SchwarzBC_Base(Teuchos::ParameterList& p);

  template <typename T>
  void
  computeBCs(size_t const ns_node, T& x_val, T& y_val, T& z_val);

#if defined(ALBANY_DTK)
  Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
  computeBCsDTK();
#endif  // ALBANY_DTK

  void
  setCoupledAppName(std::string const& can)
  {
    coupled_app_name_ = can;
  }

  std::string
  getCoupledAppName() const
  {
    return coupled_app_name_;
  }

  void
  setCoupledBlockName(std::string const& cbn)
  {
    coupled_block_name_ = cbn;
  }

  std::string
  getCoupledBlockName() const
  {
    return coupled_block_name_;
  }

  void
  setThisAppIndex(int const tai)
  {
    this_app_index_ = tai;
  }

  int
  getThisAppIndex() const
  {
    return this_app_index_;
  }

  void
  setCoupledAppIndex(int const cai)
  {
    coupled_app_index_ = cai;
  }

  int
  getCoupledAppIndex() const
  {
    return coupled_app_index_;
  }

  Albany::Application const&
  getApplication(int const app_index)
  {
    return *(coupled_apps_[app_index]);
  }

  Albany::Application const&
  getApplication(int const app_index) const
  {
    return *(coupled_apps_[app_index]);
  }

  template <typename SBC, typename T>
  friend void
  fillResidual(SBC& sbc, typename T::EvalData d);

 protected:
  Teuchos::RCP<Albany::Application> app_;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> coupled_apps_;

  std::string coupled_app_name_{"SELF"};

  std::string coupled_block_name_{"NONE"};

  int this_app_index_{-1};

  int coupled_app_index_{-1};
};

//
// Fill residual, used in both residual and Jacobian
//
template <typename SchwarzBC, typename Traits>
void
fillResidual(SchwarzBC& sbc, typename Traits::EvalData d);

//
// Residual
//
template <typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::Residual, Traits>
    : public SchwarzBC_Base<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  SchwarzBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

//
// Jacobian
//
template <typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public SchwarzBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  SchwarzBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

//
// Tangent
//
template <typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::Tangent, Traits>
    : public SchwarzBC_Base<PHAL::AlbanyTraits::Tangent, Traits>
{
 public:
  SchwarzBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

//
// Distributed Parameter Derivative
//
template <typename Traits>
class SchwarzBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public SchwarzBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
 public:
  SchwarzBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif  // LCM_SchwarzBC_hpp
