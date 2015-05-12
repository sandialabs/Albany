//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SCATTER_RESIDUAL_HPP
#define PHAL_SCATTER_RESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

namespace PHAL {
/** \brief Scatters result from the residual fields into the
    global (epetra) data structurs.  This includes the
    post-processing of the AD data type for all evaluation
    types besides Residual.

*/
// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************

template<typename EvalT, typename Traits>
class ScatterResidualBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ScatterResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d)=0;

  //Kokkos::View<int***, PHX::Device> Index;

protected:

  typedef typename EvalT::ScalarT ScalarT;
  Teuchos::RCP<PHX::FieldTag> scatter_operation;
  std::vector< PHX::MDField<ScalarT,Cell,Node> > val;
  PHX::MDField<ScalarT,Cell,Node,Dim>  valVec;
  //typedef Kokkos::View < ScalarT***, Kokkos::LayoutRight, PHX::Device > temp_view_type;
  std::vector< PHX::MDField<ScalarT,Cell,Node,Dim,Dim> > valTensor;
  std::size_t numNodes;
  std::size_t numFieldsBase; // Number of fields gathered in this call
  std::size_t offset; // Offset of first DOF being gathered when numFields<neq

  unsigned short int tensorRank;
};

template<typename EvalT, typename Traits> class ScatterResidual;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  const std::size_t numFields;

//Kokkos
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

  Teuchos::RCP<Tpetra_Vector> fT;
  Teuchos::ArrayRCP<ST> f_nonconstView;

  Kokkos::View<int***, PHX::Device> Index;

  struct ScatterRank0_Tag{};
  struct ScatterRank1_Tag{};
  struct ScatterRank2_Tag{};

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterRank0_Tag> ScatterRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterRank1_Tag> ScatterRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterRank2_Tag> ScatterRank2_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterRank0_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterRank1_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterRank2_Tag& tag, const int& i) const;

#endif
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  //typedef Kokkos::View < ScalarT***, Kokkos::LayoutRight, PHX::Device > temp_view_type;
  const std::size_t numFields;

//Kokkos
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

  Teuchos::RCP<Tpetra_Vector> fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT;

  Kokkos::View<int***, PHX::Device> Index;
  bool loadResid;
  //LO *colT;
  int neq, nunk, numDim;

  typedef typename Tpetra_CrsMatrix::local_matrix_type  LocalMatrixType;
  LocalMatrixType jacobian;

  struct ScatterRank0_is_adjoint_Tag{};
  struct ScatterRank0_no_adjoint_Tag{};
  struct ScatterRank1_is_adjoint_Tag{};
  struct ScatterRank1_no_adjoint_Tag{};
  struct ScatterRank2_is_adjoint_Tag{};
  struct ScatterRank2_no_adjoint_Tag{};


  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterRank0_is_adjoint_Tag> ScatterRank0_is_adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterRank0_no_adjoint_Tag> ScatterRank0_no_adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterRank1_is_adjoint_Tag> ScatterRank1_is_adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterRank1_no_adjoint_Tag> ScatterRank1_no_adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterRank2_is_adjoint_Tag> ScatterRank2_is_adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, ScatterRank2_no_adjoint_Tag> ScatterRank2_no_adjoint_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterRank0_is_adjoint_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterRank0_no_adjoint_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterRank1_is_adjoint_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterRank1_no_adjoint_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterRank2_is_adjoint_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterRank2_no_adjoint_Tag& tag, const int& i) const;
#endif
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Tangent,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Distributed parameter derivative
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG_MP
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::SGResidual,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::SGResidual, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::SGJacobian,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::SGJacobian, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::SGTangent,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::SGTangent, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::MPResidual,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::MPJacobian,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::MPJacobian, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::MPTangent,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::MPTangent, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  const std::size_t numFields;
};
#endif //ALBANY_SG_MP

// **************************************************************
}

#endif
