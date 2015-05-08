//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_RESPONSESURFACEVELOCITYMISMATCH_HPP
#define FELIX_RESPONSESURFACEVELOCITYMISMATCH_HPP

//#include "FELIX_MeshRegion.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace FELIX {
/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseSurfaceVelocityMismatch :
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    
    ResponseSurfaceVelocityMismatch(Teuchos::ParameterList& p,
			 const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
				     PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    int  cellDims, sideDims, numQPsSide, numNodes;

    std::size_t numQPs;
    std::size_t numDims;
    std::size_t numVecDim;
    
    PHX::MDField<ScalarT,Cell,Node,VecDim> velocity_field;
    PHX::MDField<ScalarT,Cell,Node,VecDim> surfaceVelocity_field;
    PHX::MDField<ScalarT,Cell,Node,VecDim> velocityRMS_field;
    PHX::MDField<ScalarT,Cell,Node> basal_friction_field;
    PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;

    Teuchos::RCP<shards::CellTopology> cellType;
    Teuchos::RCP<shards::CellTopology> sideType;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubatureCell;
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubatureSide;

    // The basis
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;

    // Temporary FieldContainers
    Intrepid::FieldContainer<RealType> cubPointsSide;
    //const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs;
    Intrepid::FieldContainer<RealType> refPointsSide;
    Intrepid::FieldContainer<RealType> cubWeightsSide;
    Intrepid::FieldContainer<MeshScalarT> physPointsSide;
    Intrepid::FieldContainer<MeshScalarT> jacobianSide;
    Intrepid::FieldContainer<MeshScalarT> invJacobianSide;
    Intrepid::FieldContainer<MeshScalarT> jacobianSide_det;

    Intrepid::FieldContainer<MeshScalarT> physPointsCell;

    Intrepid::FieldContainer<MeshScalarT> weighted_measure;
    Intrepid::FieldContainer<RealType> basis_refPointsSide;
    Intrepid::FieldContainer<RealType> basisGrad_refPointsSide;
    Intrepid::FieldContainer<MeshScalarT> trans_basis_refPointsSide;
    Intrepid::FieldContainer<MeshScalarT> trans_gradBasis_refPointsSide;
    Intrepid::FieldContainer<MeshScalarT> weighted_trans_basis_refPointsSide;

    Intrepid::FieldContainer<ScalarT> dofCell;
    Intrepid::FieldContainer<ScalarT> dofSide;

    Intrepid::FieldContainer<ScalarT> dofCellVec;
    Intrepid::FieldContainer<ScalarT> dofSideVec;

    Intrepid::FieldContainer<ScalarT> data;

    std::string sideSetID;
    Teuchos::Array<RealType> inputValues;
    ScalarT p_resp, p_reg, resp, reg;
    double scaling, alpha;
  };
	
}

#endif
