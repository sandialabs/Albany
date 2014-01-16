//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SCATTER_RESIDUAL_HPP
#define PHAL_SCATTER_RESIDUAL_HPP

#include "Phalanx_ConfigDefs.hpp"
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

protected:

  typedef typename EvalT::ScalarT ScalarT;
  Teuchos::RCP<PHX::FieldTag> scatter_operation;
  std::vector< PHX::MDField<ScalarT,Cell,Node> > val;
  std::vector< PHX::MDField<ScalarT,Cell,Node,Dim> > valVec;
  std::size_t numNodes;
  std::size_t numFieldsBase; // Number of fields gathered in this call
  std::size_t offset; // Offset of first DOF being gathered when numFields<neq

  bool vectorField;
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
  const std::size_t numFields;
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
