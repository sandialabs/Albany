//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADAPT_ELEMENTSIZE_HPP
#define ADAPT_ELEMENTSIZE_HPP

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_ProblemUtils.hpp"

namespace Adapt {
/** 
 * \brief Description
 */
  template<typename EvalT, typename Traits>
  class ElementSizeFieldBase : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    ElementSizeFieldBase(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);
    
    // These functions are defined in the specializations
    void preEvaluate(typename Traits::PreEvalData d) = 0;
    void postEvaluate(typename Traits::PostEvalData d) = 0;
    void evaluateFields(typename Traits::EvalData d) = 0;

    Teuchos::RCP<const PHX::FieldTag> getEvaluatedFieldTag() const {
      return size_field_tag;
    }

    Teuchos::RCP<const PHX::FieldTag> getResponseFieldTag() const {
      return size_field_tag;
    }
    
  protected:

    typedef enum {NOTSCALED, SCALAR, VECTOR} ScalingType;

    Teuchos::RCP<const Teuchos::ParameterList> getValidSizeFieldParameters() const;

    void getCellRadius(const std::size_t cell, MeshScalarT& cellRadius) const;

    std::string scalingName;
    std::string className;
    
    std::size_t numQPs;
    std::size_t numDims;
    std::size_t numVertices;
    
    PHX::MDField<MeshScalarT,Cell,QuadPoint> qp_weights;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<MeshScalarT,Cell,Node,Dim> coordVec_vertices; //not currently needed
//    PHX::MDField<ScalarT> field;

    bool outputToExodus;
    bool outputCellAverage;
    bool outputQPData;
    bool outputNodeData;
    bool isAnisotropic;
    ScalingType scalingType;

    std::string vectorOp;

    Teuchos::RCP< PHX::Tag<ScalarT> > size_field_tag;
    Albany::StateManager* pStateMgr;

  };

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits> class ElementSizeField;

// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class ElementSizeField<PHAL::AlbanyTraits::Residual,Traits>
   : public ElementSizeFieldBase<PHAL::AlbanyTraits::Residual, Traits> {
public:
  ElementSizeField(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ElementSizeField<PHAL::AlbanyTraits::Jacobian,Traits>
   : public ElementSizeFieldBase<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  ElementSizeField(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ElementSizeField<PHAL::AlbanyTraits::Tangent,Traits>
   : public ElementSizeFieldBase<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  ElementSizeField(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************
#ifdef ALBANY_SG_MP
template<typename Traits>
class ElementSizeField<PHAL::AlbanyTraits::SGResidual,Traits>
   : public ElementSizeFieldBase<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  ElementSizeField(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class ElementSizeField<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public ElementSizeFieldBase<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  ElementSizeField(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class ElementSizeField<PHAL::AlbanyTraits::SGTangent,Traits>
   : public ElementSizeFieldBase<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  ElementSizeField(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class ElementSizeField<PHAL::AlbanyTraits::MPResidual,Traits>
   : public ElementSizeFieldBase<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  ElementSizeField(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class ElementSizeField<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public ElementSizeFieldBase<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  ElementSizeField(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class ElementSizeField<PHAL::AlbanyTraits::MPTangent,Traits>
   : public ElementSizeFieldBase<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  ElementSizeField(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};
#endif //ALBANY_SG_MP

}

#endif  // Adapt_ElementSizeField.hpp
