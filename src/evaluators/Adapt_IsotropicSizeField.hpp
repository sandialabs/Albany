//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ADAPT_ISOTROPICSIZE_HPP
#define ADAPT_ISOTROPICSIZE_HPP

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
  class IsotropicSizeField : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    IsotropicSizeField(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);
    
    void evaluateFields(typename Traits::EvalData d);

    Teuchos::RCP<const PHX::FieldTag> getEvaluatedFieldTag() const {
      return size_field_tag;
    }

    Teuchos::RCP<const PHX::FieldTag> getResponseFieldTag() const {
      return size_field_tag;
    }
    
  private:

    typedef enum {NOTSCALED, SCALAR, VECTOR} ScalingType;

    Teuchos::RCP<const Teuchos::ParameterList> getValidSizeFieldParameters() const;

    void getCellRadius(const std::size_t cell, ScalarT& cellRadius) const;

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
    ScalingType scalingType;

    std::string vectorOp;

    Teuchos::RCP< PHX::Tag<ScalarT> > size_field_tag;
  };

	
}

#endif  // Adapt_IsotropicSizeField.hpp
