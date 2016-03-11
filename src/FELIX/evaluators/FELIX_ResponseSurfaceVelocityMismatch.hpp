//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_RESPONSESURFACEVELOCITYMISMATCH_HPP
#define FELIX_RESPONSESURFACEVELOCITYMISMATCH_HPP

//#include "FELIX_MeshRegion.hpp"
#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

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
    typedef typename EvalT::ParamScalarT ParamScalarT;
    
    ResponseSurfaceVelocityMismatch(Teuchos::ParameterList& p,
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
    std::size_t numVecDim;
    
    PHX::MDField<ScalarT,Cell,Node,VecDim> velocity_field;
    PHX::MDField<ParamScalarT,Cell,Node,VecDim> surfaceVelocity_field;
    PHX::MDField<ParamScalarT,Cell,Node,VecDim> velocityRMS_field;
    PHX::MDField<ParamScalarT,Cell,Node> basal_friction_field;
    PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;

    Teuchos::RCP<shards::CellTopology> cellType;
    Teuchos::RCP<shards::CellTopology> sideType;
    Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubatureCell;
    Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubatureSide;

    // The basis
    Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis;

    // Temporary FieldContainers
    Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> cubPointsSide;
    //const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs;
    Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> refPointsSide;
    Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> cubWeightsSide;
    Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> physPointsSide;
    Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobianSide;
    Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> invJacobianSide;
    Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobianSide_det;

    Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> physPointsCell;

    Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> weighted_measure;
    Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> basis_refPointsSide;
    Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> basisGrad_refPointsSide;
    Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> trans_basis_refPointsSide;
    Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> trans_gradBasis_refPointsSide;
    Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> weighted_trans_basis_refPointsSide;

    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofCell;
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofSide;

    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofCellVec;
    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofSideVec;

    Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> data;

    std::string sideSetID;
    Teuchos::Array<RealType> inputValues;
    ScalarT p_resp, p_reg, resp, reg;
    double scaling, alpha, asinh_scaling;
  };
	
}

#endif
