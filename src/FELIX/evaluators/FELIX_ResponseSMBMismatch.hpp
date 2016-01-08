//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_RESPONSESMBMISMATCH_HPP
#define FELIX_RESPONSESMBMISMATCH_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

namespace FELIX {
/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseSMBMismatch :
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    
    ResponseSMBMismatch(Teuchos::ParameterList& p,
			 const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
				     PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    int  cellDims, sideDims, numQPsSide, numNodes, cubatureDegree;

    std::size_t numQPs;
    std::size_t numDims;
    std::size_t numVecFODims;
    

    PHX::MDField<ScalarT,Cell,Node> H;
    PHX::MDField<ScalarT,Cell,Node,VecDim> velocity_field;
    PHX::MDField<ScalarT,Cell,Node> SMB;
    PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;

    Teuchos::RCP<shards::CellTopology> cellType;
    Teuchos::RCP<shards::CellTopology> sideType;
    Teuchos::RCP<Intrepid2::Cubature<RealType> > cubatureCell;
    Teuchos::RCP<Intrepid2::Cubature<RealType> > cubatureSide;

    // The basis
    Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer<RealType> > > intrepidBasis;

    // Temporary FieldContainers
    Intrepid2::FieldContainer<RealType> cubPointsSide;
    //const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs;
    Intrepid2::FieldContainer<RealType> refPointsSide;
    Intrepid2::FieldContainer<RealType> cubWeightsSide;
    Intrepid2::FieldContainer<MeshScalarT> physPointsSide;
    Intrepid2::FieldContainer<MeshScalarT> jacobianSide;
    Intrepid2::FieldContainer<MeshScalarT> invJacobianSide;
    Intrepid2::FieldContainer<MeshScalarT> jacobianSide_det;

    Intrepid2::FieldContainer<MeshScalarT> physPointsCell;

    Intrepid2::FieldContainer<MeshScalarT> weighted_measure;
    Intrepid2::FieldContainer<RealType> basis_refPointsSide;
    Intrepid2::FieldContainer<RealType> basisGrad_refPointsSide;
    Intrepid2::FieldContainer<MeshScalarT> trans_basis_refPointsSide;
    Intrepid2::FieldContainer<MeshScalarT> trans_gradBasis_refPointsSide;
    Intrepid2::FieldContainer<MeshScalarT> weighted_trans_basis_refPointsSide;

    Intrepid2::FieldContainer<ScalarT> dofCell;
    Intrepid2::FieldContainer<ScalarT> dofSide;

    Intrepid2::FieldContainer<ScalarT> dofCellVec;
    Intrepid2::FieldContainer<ScalarT> dofSideVec;

    Intrepid2::FieldContainer<ScalarT> data;

    std::string sideSetID;
    Teuchos::Array<RealType> inputValues;
    ScalarT p_resp, p_reg, resp, reg;
    double scaling, alpha, asinh_scaling;
  };
	
}

#endif
