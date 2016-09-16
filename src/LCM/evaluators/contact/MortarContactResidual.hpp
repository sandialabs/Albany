//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MORTAR_CONTACT_RESIDUAL_HPP
#define MORTAR_CONTACT_RESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

namespace LCM {
/** \brief This class implements the Mortar contact algorithm. Here is the overall sketch of how things work:

   General context: A workset of elements are processed to assemble local finite element residual contributions that
   the opposite contacting surface will impose on the current workset of elements.

   1. Do a global search to find all the slave segments that can potentially intersect the master segments that this
      processor owns. This is done in preEvaluate, as we don't want to loop over worksets and we want to do the global
      search once per processor.

   2. For the elements in the workset, find the element surfaces that are master surface segments. In the beginning of
      evaluate, do a local search to find the slave segments that potentially intersect each master segment. Note that this
      can change each evaluate call (Newton iteration).

   3. In evaluate, form the mortar integration space and assemble all the slave constraint contributions into the master side
      locations residual vector - ultimately the elements of the current workset.

*/
// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************

template<typename EvalT, typename Traits>
class MortarContactResidualBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  MortarContactResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

// These functions are defined in the specializations
  virtual void preEvaluate(typename Traits::PreEvalData d) = 0;
  virtual void evaluateFields(typename Traits::EvalData d) = 0;

protected:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  Teuchos::RCP<PHX::FieldTag> mortar_projection_operation;
  std::vector< PHX::MDField<ScalarT,Cell,Node>> val;
  std::vector< PHX::MDField<ScalarT,Cell,Node,Dim>> valVec;
  std::vector< PHX::MDField<ScalarT,Cell,Node,Dim,Dim>> valTensor;
  std::size_t numNodes;
  std::size_t numFieldsBase; // Number of fields gathered in this call
  std::size_t offset; // Offset of first DOF being gathered when numFields<neq

  const Teuchos::ArrayRCP<std::string>& masterSideNames;
  const Teuchos::ArrayRCP<std::string>& slaveSideNames;
  const Teuchos::ArrayRCP<std::string>& sideSetIDs;
  const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs;

//! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;


  unsigned short int tensorRank;
};

template<typename EvalT, typename Traits> class MortarContactResidual;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::Residual,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::Jacobian,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::Tangent,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Distributed parameter derivative
// **************************************************************
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::SGResidual,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::SGResidual, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::SGJacobian,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::SGJacobian, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::SGTangent,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::SGTangent, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  const std::size_t numFields;
};
#endif 
#ifdef ALBANY_ENSEMBLE 

// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::MPResidual,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::MPJacobian,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::MPJacobian, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class MortarContactResidual<PHAL::AlbanyTraits::MPTangent,Traits>
  : public MortarContactResidualBase<PHAL::AlbanyTraits::MPTangent, Traits>  {
public:
  MortarContactResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  const std::size_t numFields;
};
#endif

// **************************************************************
}

#endif
