//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_RESPONSESADDLEVALUE_HPP
#define QCAD_RESPONSESADDLEVALUE_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "QCAD_EvaluatorTools.hpp"
#include "QCAD_SaddleValueResponseFunction.hpp"
#include "QCAD_MaterialDatabase.hpp"


/** 
 * \brief Response Description
 */
namespace QCAD 
{
  template<typename EvalT, typename Traits>
  class ResponseSaddleValue : 
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>,
    public EvaluatorTools<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    
    ResponseSaddleValue(Teuchos::ParameterList& p,
			const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
				     PHX::FieldManager<Traits>& vm);
  
    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);

    /*int numResponses() const {  // I don't think this is needed anymore (egn)
      return 5; 
      }*/
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;
    void getCellQuantities(const std::size_t cell, ScalarT& cellVol, 
			   typename EvalT::ScalarT& fieldVal, typename EvalT::ScalarT& retFieldVal, 
			   std::vector<typename EvalT::ScalarT>& fieldGrad) const;
    void getCellArea(const std::size_t cell, typename EvalT::ScalarT& cellArea) const;
    void getAvgCellCoordinates(PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec,
			       const std::size_t cell, double* dblAvgCoords, double& dblMaxZ) const;

    std::size_t numQPs;
    std::size_t numDims;
    std::size_t numVertices;
  
    Teuchos::RCP<QCAD::SaddleValueResponseFunction> svResponseFn;
  
    PHX::MDField<const ScalarT> field;
    PHX::MDField<const ScalarT> fieldGradient;
    PHX::MDField<const ScalarT> retField;
    PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<const MeshScalarT,Cell,Node,Dim> coordVec_vertices; //not currently needed
    PHX::MDField<const MeshScalarT,Cell,QuadPoint> weights;
    
    std::string fieldName;
    std::string fieldGradientName;
    std::string retFieldName;
    
    bool bReturnSameField;
    double scaling, gradScaling, retScaling;
    double lattTemp; 
    
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

  };
	
}

#endif
