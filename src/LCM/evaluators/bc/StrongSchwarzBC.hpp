//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_StrongSchwarzBC_hpp)
#define LCM_StrongSchwarzBC_hpp

#include "Albany_config.h"

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "Sacado_ParameterAccessor.hpp"
//#include "PHAL_Dirichlet.hpp"
#include "PHAL_SDirichlet.hpp"

#if defined(ALBANY_DTK)
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "DTK_MapOperatorFactory.hpp"
#include "DTK_STKMeshHelpers.hpp"
#include "DTK_STKMeshManager.hpp"
#endif

namespace LCM {

///
/// StrongSchwarz BC evaluator
///

//
// Specialization of the DirichletBase class
//
template <typename EvalT, typename Traits>
class StrongSchwarzBC;

template <typename EvalT, typename Traits>
// class StrongSchwarzBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
class StrongSchwarzBC_Base : public PHAL::SDirichlet<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;

  StrongSchwarzBC_Base(Teuchos::ParameterList& p);

  template <typename T>
  void
  computeBCs(size_t const ns_node, T& x_val, T& y_val, T& z_val);

#if defined(ALBANY_DTK)
  Teuchos::Array<Teuchos::RCP<
      Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>>
  computeBCsDTK();

  Teuchos::RCP<Tpetra::MultiVector<double, int, DataTransferKit::SupportId>>
  doDTKInterpolation(
      DataTransferKit::STKMeshManager&                    coupled_manager,
      DataTransferKit::STKMeshManager&                    this_manager,
      Albany::AbstractSTKFieldContainer::VectorFieldType* coupled_field,
      Albany::AbstractSTKFieldContainer::VectorFieldType* this_field,
      const int                                           neq,
      Teuchos::ParameterList&                             dtk_params);
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
  fillSolution(SBC& sbc, typename T::EvalData d);

  template <typename SBC, typename T>
  friend void
  fillResidual(SBC& sbc, typename T::EvalData d);

 protected:
  Teuchos::RCP<Albany::Application> app_{Teuchos::null};

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application>> coupled_apps_;

  std::string coupled_app_name_{"SELF"};

  std::string coupled_block_name_{"NONE"};

  int this_app_index_{-1};

  int coupled_app_index_{-1};
};

//
// Fill solution with Dirichlet values 
//
template <typename StrongSchwarzBC, typename Traits>
void
fillSolution(StrongSchwarzBC& sbc, typename Traits::EvalData d);

//
// Fill residual, used in both residual and Jacobian
//
template <typename StrongSchwarzBC, typename Traits>
void
fillResidual(StrongSchwarzBC& sbc, typename Traits::EvalData d);

//
// Residual
//
template <typename Traits>
class StrongSchwarzBC<PHAL::AlbanyTraits::Residual, Traits>
    : public StrongSchwarzBC_Base<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  StrongSchwarzBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void
  preEvaluate(typename Traits::EvalData d);
  void
  evaluateFields(typename Traits::EvalData d);
};

//
// Jacobian
//
template <typename Traits>
class StrongSchwarzBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public StrongSchwarzBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  StrongSchwarzBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

//
// Tangent
//
template <typename Traits>
class StrongSchwarzBC<PHAL::AlbanyTraits::Tangent, Traits>
    : public StrongSchwarzBC_Base<PHAL::AlbanyTraits::Tangent, Traits>
{
 public:
  StrongSchwarzBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

//
// Distributed Parameter Derivative
//
template <typename Traits>
class StrongSchwarzBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public StrongSchwarzBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
 public:
  StrongSchwarzBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif  // LCM_StrongSchwarzBC_hpp
