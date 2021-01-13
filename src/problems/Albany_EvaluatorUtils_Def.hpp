//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "PHAL_ComputeBasisFunctions.hpp"
#include "PHAL_ComputeBasisFunctionsSide.hpp"
#include "PHAL_DOFCellToSide.hpp"
#include "PHAL_DOFCellToSideQP.hpp"
#include "PHAL_DOFGradInterpolation.hpp"
#include "PHAL_DOFGradInterpolationSide.hpp"
#include "PHAL_DOFInterpolation.hpp"
#include "PHAL_DOFInterpolationSide.hpp"
#include "PHAL_DOFSideToCell.hpp"
#include "PHAL_DOFTensorGradInterpolation.hpp"
#include "PHAL_DOFTensorInterpolation.hpp"
#include "PHAL_DOFVecGradInterpolation.hpp"
#include "PHAL_DOFVecGradInterpolationSide.hpp"
#include "PHAL_DOFVecInterpolation.hpp"
#include "PHAL_DOFVecInterpolationSide.hpp"
#include "PHAL_GatherSolution.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_MapToPhysicalFrame.hpp"
#include "PHAL_MapToPhysicalFrameSide.hpp"
#include "PHAL_P0Interpolation.hpp"
#include "PHAL_ScatterResidual.hpp"
#include "PHAL_ScatterSideEqnResidual.hpp"
#include "PHAL_ScatterScalarNodalParameter.hpp"

namespace Albany {

/********************  Problem Utils Class  ******************************/

template<typename EvalT, typename Traits, typename ScalarType>
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::EvaluatorUtilsImpl(
     Teuchos::RCP<Layouts> dl_) :
     dl(dl_)
{
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructGatherSolutionEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    if(isVectorField)
      p->set<int>("Tensor Rank", 1);
    else
      p->set<int>("Tensor Rank", 0);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    p->set< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names", dof_names_dot);
    return rcp(new PHAL::GatherSolution<EvalT,Traits>(*p,dl));
}


template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructGatherSolutionEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    p->set<int>("Tensor Rank", tensorRank);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    p->set< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names", dof_names_dot);
    return rcp(new PHAL::GatherSolution<EvalT,Traits>(*p,dl));
}


template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructGatherSolutionEvaluator_withAcceleration(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    if(isVectorField)
      p->set<int>("Tensor Rank", 1);
    else
      p->set<int>("Tensor Rank", 0);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    if (dof_names_dot != Teuchos::null)
      p->set< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names", dof_names_dot);
    else
      p->set<bool>("Disable Transient", true);

    if (dof_names_dotdot != Teuchos::null) {
      p->set< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names", dof_names_dotdot);
      p->set<bool>("Enable Acceleration", true);
    }

    return rcp(new PHAL::GatherSolution<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructGatherSolutionEvaluator_withAcceleration(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    p->set<int>("Tensor Rank", tensorRank);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    if (dof_names_dot != Teuchos::null)
      p->set< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names", dof_names_dot);
    else
      p->set<bool>("Disable Transient", true);

    if (dof_names_dotdot != Teuchos::null) {
      p->set< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names", dof_names_dotdot);
      p->set<bool>("Enable Acceleration", true);
    }

    return rcp(new PHAL::GatherSolution<EvalT,Traits>(*p,dl));
}


template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructGatherSolutionEvaluator_noTransient(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    if(isVectorField)
      p->set<int>("Tensor Rank", 1);
    else
      p->set<int>("Tensor Rank", 0);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<bool>("Disable Transient", true);

    return rcp(new PHAL::GatherSolution<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructGatherSolutionEvaluator_noTransient(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    p->set<int>("Tensor Rank", tensorRank);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<bool>("Disable Transient", true);

    return rcp(new PHAL::GatherSolution<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructGatherScalarNodalParameter(
       const std::string& param_name,
       const std::string& field_name) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Parameter"));
    p->set<std::string>("Parameter Name", param_name);
    if (field_name!="")
      p->set<std::string>("Field Name", field_name);
    else
      p->set<std::string>("Field Name", param_name);

    return rcp(new PHAL::GatherScalarNodalParameter<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructScatterScalarNodalParameter(
       const std::string& param_name,
       const std::string& field_name) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Scatter Parameter"));
    p->set<std::string>("Parameter Name", param_name);
    if (field_name!="") {
      p->set<std::string>("Field Name", field_name);
    } else {
      p->set<std::string>("Field Name", param_name);
    }

    return rcp(new PHAL::ScatterScalarNodalParameter<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructGatherScalarExtruded2DNodalParameter(
       const std::string& param_name,
       const std::string& field_name) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Parameter"));
    p->set<std::string>("Parameter Name", param_name);
    if (field_name!="")
      p->set<std::string>("Field Name", field_name);
    else
      p->set<std::string>("Field Name", param_name);

    p->set<int>("Field Level", 0);
    return rcp(new PHAL::GatherScalarExtruded2DNodalParameter<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructScatterScalarExtruded2DNodalParameter(
       const std::string& param_name,
       const std::string& field_name) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Scatter Parameter"));
    p->set<std::string>("Parameter Name", param_name);
    if (field_name!="") {
      p->set<std::string>("Field Name", field_name);
    } else {
      p->set<std::string>("Field Name", param_name);
    }

    p->set<int>("Field Level", 0);
    return rcp(new PHAL::ScatterScalarExtruded2DNodalParameter<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructScatterResidualEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF, std::string scatterName) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Scatter Residual"));
    p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);

    if(isVectorField)
      p->set<int>("Tensor Rank", 1);
    else
      p->set<int>("Tensor Rank", 0);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<std::string>("Scatter Field Name", scatterName);

    return rcp(new PHAL::ScatterResidual<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructScatterSideEqnResidualEvaluator(
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& sideSetName,
       bool residualsAreVolumeFields,
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF, std::string scatterName) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Scatter Residual"));
    p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);

    if(isVectorField)
      p->set<int>("Tensor Rank", 1);
    else
      p->set<int>("Tensor Rank", 0);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name", sideSetName);
    p->set<bool>("Residuals Are Volume Fields", residualsAreVolumeFields);
    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<std::string>("Scatter Field Name", scatterName);

    return rcp(new PHAL::ScatterSideEqnResidual<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructScatterResidualEvaluatorWithExtrudedParams(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       Teuchos::RCP<std::map<std::string, int> > extruded_params_levels,
       int offsetToFirstDOF, std::string scatterName) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Scatter Residual"));
    p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);
    p->set< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels", extruded_params_levels);

    if(isVectorField)
      p->set<int>("Tensor Rank", 1);
    else
      p->set<int>("Tensor Rank", 0);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<std::string>("Scatter Field Name", scatterName);

    return rcp(new PHAL::ScatterResidualWithExtrudedParams<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructScatterResidualEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF, std::string scatterName) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Scatter Residual"));
    p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);

    p->set<int>("Tensor Rank", tensorRank);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<std::string>("Scatter Field Name", scatterName);

    return rcp(new PHAL::ScatterResidual<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructScatterSideEqnResidualEvaluator(
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& sideSetName,
       bool residualsAreVolumeFields,
       int tensorRank,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF, std::string scatterName) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Scatter Residual"));
    p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);

    p->set<int>("Tensor Rank", tensorRank);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name", sideSetName);
    p->set<bool>("Residuals Are Volume Fields", residualsAreVolumeFields);
    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<std::string>("Scatter Field Name", scatterName);

    return rcp(new PHAL::ScatterSideEqnResidual<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructGatherCoordinateVectorEvaluator(
    std::string strCurrentDisp) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Coordinate Vector"));

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);

    // Output:: Coordindate Vector at vertices
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    if( strCurrentDisp != "" )
      p->set<std::string>("Current Displacement Vector Name", strCurrentDisp);

    return rcp(new PHAL::GatherCoordinateVector<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructMapToPhysicalFrameEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<IntrepidCubature> cubature,
    const Teuchos::RCP<IntrepidBasis> intrepidBasis) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Map To Physical Frame"));

    // Input: X, Y at vertices
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<RCP <IntrepidCubature> >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set< RCP<IntrepidBasis> >
        ("Intrepid2 Basis", intrepidBasis);

    // Output: X, Y at Quad Points (same name as input)

    return rcp(new PHAL::MapToPhysicalFrame<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructMapToPhysicalFrameSideEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<IntrepidCubature> cubature,
    const std::string& sideSetName) const
{
    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! The layout structure for side set " << sideSetName << " was not found.\n");

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("Map To Physical Frame Side"));

    // Input: X, Y at vertices
    p->set<std::string>("Coordinate Vector Vertex Name", "Coord Vec " + sideSetName);
    p->set<std::string>("Coordinate Vector QP Name", "Coord Vec " + sideSetName);
    p->set< RCP<IntrepidCubature> >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name", sideSetName);

    // Output: X, Y at Quad Points (same name as input)
    return rcp(new PHAL::MapToPhysicalFrameSide<EvalT,Traits>(*p,dl->side_layouts.at(sideSetName)));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructComputeBasisFunctionsEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<IntrepidBasis> intrepidBasis,
    const Teuchos::RCP<IntrepidCubature> cubature) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name",coord_vec_name);
    p->set< RCP<IntrepidCubature> >("Cubature", cubature);

    p->set< RCP<IntrepidBasis> >
        ("Intrepid2 Basis", intrepidBasis);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<std::string>("Weights Name",              weights_name);
    p->set<std::string>("Jacobian Det Name",         jacobian_det_name);
    p->set<std::string>("Jacobian Name",             jacobian_det_name);
    p->set<std::string>("Jacobian Inv Name",         jacobian_inv_name);
    p->set<std::string>("BF Name",                   bf_name);
    p->set<std::string>("Weighted BF Name",          weighted_bf_name);
    p->set<std::string>("Gradient BF Name",          grad_bf_name);
    p->set<std::string>("Weighted Gradient BF Name", weighted_grad_bf_name);

    return rcp(new PHAL::ComputeBasisFunctions<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructComputeBasisFunctionsSideEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<IntrepidBasis> intrepidBasisSide,
    const Teuchos::RCP<IntrepidCubature> cubatureSide,
    const std::string& sideSetName,
    const bool buildNormals,
    const bool planar) const
{
    TEUCHOS_TEST_FOR_EXCEPTION (buildNormals && planar, std::runtime_error,
                                "Error! Cannot compute normal for planar sufaces at the moment.\n");

    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! The layout structure for side set " << sideSetName << " was not found.\n");

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    std::string sideSetName_ = planar ? sideSetName + "_planar" : sideSetName;

    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions Side"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<std::string>("Side Coordinate Vector Name",coord_vec_name + " " + sideSetName);
    p->set< RCP<IntrepidCubature> >("Cubature Side", cubatureSide);
    p->set< RCP<IntrepidBasis> >("Intrepid Basis Side", intrepidBasisSide);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name",sideSetName);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<std::string>("Weighted Measure Name",     weighted_measure_name + " "+sideSetName_);
    p->set<std::string>("Tangents Name",             tangents_name + " "+sideSetName_);
    p->set<std::string>("Metric Name",               metric_name + " "+sideSetName_);
    p->set<std::string>("Metric Determinant Name",   metric_det_name + " "+sideSetName_);
    p->set<std::string>("BF Name",                   bf_name + " "+sideSetName_);
    p->set<std::string>("Gradient BF Name",          grad_bf_name + " "+sideSetName_);
    p->set<std::string>("Inverse Metric Name",       metric_inv_name + " "+sideSetName_);
    if (buildNormals) {
      p->set<std::string>("Side Normal Name",normal_name + " " + sideSetName_);
      p->set<std::string>("Coordinate Vector Name",coord_vec_name);
    }

    p->set("Side Set Is Planar",planar);

    return rcp(new PHAL::ComputeBasisFunctionsSide<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFCellToSideEvaluator(
    const std::string& cell_dof_name,
    const std::string& sideSetName,
    const std::string& layout,
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const std::string& side_dof_name) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Cell To Side"));

    // Input
    p->set<std::string>("Cell Variable Name", cell_dof_name);
    p->set<std::string>("Data Layout", layout);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name", sideSetName);

    // Output
    if (side_dof_name!="")
      p->set<std::string>("Side Variable Name", side_dof_name);
    else
      p->set<std::string>("Side Variable Name", cell_dof_name);

    return rcp(new PHAL::DOFCellToSideBase<EvalT,Traits,ScalarType>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFCellToSideQPEvaluator(
       const std::string& cell_dof_name,
       const std::string& sideSetName,
       const std::string& layout,
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& side_dof_name) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Cell To Side"));

    // Input
    p->set<std::string>("Cell Variable Name", cell_dof_name);
    p->set<std::string>("Data Layout", layout);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("BF Name", "BF "+sideSetName);
    p->set<std::string>("Side Set Name", sideSetName);

    // Output
    if (side_dof_name!="")
      p->set<std::string>("Side Variable Name", side_dof_name);
    else
      p->set<std::string>("Side Variable Name", cell_dof_name);

    return rcp(new PHAL::DOFCellToSideQPBase<EvalT,Traits,ScalarType>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFSideToCellEvaluator(
       const std::string& side_dof_name,
       const std::string& sideSetName,
       const std::string& layout,
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& cell_dof_name) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Side To Cell"));

    // Input
    p->set<std::string>("Side Variable Name", side_dof_name);
    p->set<std::string>("Data Layout", layout);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name", sideSetName);

    // Output
    if (cell_dof_name!="")
      p->set<std::string>("Cell Variable Name", cell_dof_name);
    else
      p->set<std::string>("Cell Variable Name", side_dof_name);

    return rcp(new PHAL::DOFSideToCellBase<EvalT,Traits,ScalarType>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFGradInterpolationEvaluator(
    const std::string& dof_name, int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Grad Interpolation "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("Gradient BF Name", "Grad BF");
    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    // Output (assumes same Name as input)
    p->set<std::string>("Gradient Variable Name", dof_name+" Gradient");

    if(offsetToFirstDOF == -1)
    {
      return rcp(new PHAL::DOFGradInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
    }
    else  //works only for solution or a set of solution components
      return rcp(new PHAL::FastSolutionGradInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFGradInterpolationSideEvaluator(
       const std::string& dof_name,
       const std::string& sideSetName,
       const bool planar) const
{
    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! The layout structure for side set " << sideSetName << " was not found.\n");

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    std::string sideSetName_ = planar ? sideSetName + "_planar" : sideSetName;
    std::string gradientSuffixName = planar ? " Planar Gradient" : " Gradient";

    RCP<ParameterList> p = rcp(new ParameterList("DOF Grad Interpolation Side "+dof_name));

    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("Gradient BF Name", "Grad BF "+sideSetName_);
    p->set<std::string> ("Side Set Name",sideSetName);

    // Output (assumes same Name as input)
    p->set<std::string>("Gradient Variable Name", dof_name+gradientSuffixName);

    return rcp(new PHAL::DOFGradInterpolationSideBase<EvalT,Traits,ScalarType>(*p,dl->side_layouts.at(sideSetName)));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFInterpolationEvaluator(
    const std::string& dof_name, int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("BF Name", "BF");
    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    // Output (assumes same Name as input)

    return rcp(new PHAL::DOFInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFInterpolationSideEvaluator(
    const std::string& dof_name, const std::string& sideSetName) const
{
    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! The layout structure for side set " << sideSetName << " was not found.\n");

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation Side "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("BF Name", "BF "+sideSetName);
    p->set<std::string>("Side Set Name",sideSetName);

    // Output (assumes same Name as input)

    return rcp(new PHAL::DOFInterpolationSideBase<EvalT,Traits,ScalarType>(*p,dl->side_layouts.at(sideSetName)));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFTensorInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOFTensor Interpolation "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("BF Name", "BF");
    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    // Output (assumes same Name as input)
    if(offsetToFirstDOF == -1)
      return rcp(new PHAL::DOFTensorInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
    else  //works only for solution or a set of solution components
      return rcp(new PHAL::FastSolutionTensorInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFTensorGradInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOFTensorGrad Interpolation "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("Gradient BF Name", "Grad BF");
    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    // Output (assumes same Name as input)
    p->set<std::string>("Gradient Variable Name", dof_name+" Gradient");

    if(offsetToFirstDOF == -1)
      return rcp(new PHAL::DOFTensorGradInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
    else  //works only for solution or a set of solution components
      return rcp(new PHAL::FastSolutionTensorGradInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));

}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFVecGradInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOFVecGrad Interpolation "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("Gradient BF Name", "Grad BF");
    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    // Output (assumes same Name as input)
    p->set<std::string>("Gradient Variable Name", dof_name+" Gradient");

    if(offsetToFirstDOF == -1)
      return rcp(new PHAL::DOFVecGradInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
    else  //works only for solution or a set of solution components
      return rcp(new PHAL::FastSolutionVecGradInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFVecGradInterpolationSideEvaluator(
       const std::string& dof_name,
       const std::string& sideSetName) const
{
    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! The layout structure for side set " << sideSetName << " was not found.\n");

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Grad Interpolation Side "+dof_name));

    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("Gradient BF Name", "Grad BF "+sideSetName);
    p->set<std::string> ("Side Set Name",sideSetName);

    // Output (assumes same Name as input)
    p->set<std::string>("Gradient Variable Name", dof_name+" Gradient");

    return rcp(new PHAL::DOFVecGradInterpolationSideBase<EvalT,Traits,ScalarType>(*p,dl->side_layouts.at(sideSetName)));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFVecInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF) const
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOFVec Interpolation "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("BF Name", "BF");
    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    // Output (assumes same Name as input)
    if(offsetToFirstDOF == -1)
      return rcp(new PHAL::DOFVecInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
    else  //works only for solution or a set of solution components
      return rcp(new PHAL::FastSolutionVecInterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructDOFVecInterpolationSideEvaluator(
       const std::string& dof_name,
       const std::string& sideSetName) const
{
    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! The layout structure for side set " << sideSetName << " was not found.\n");

    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Vec Interpolation Side "+dof_name));

    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("BF Name", "BF "+sideSetName);
    p->set<std::string>("Side Set Name",sideSetName);

    // Output (assumes same Name as input)

    return rcp(new PHAL::DOFVecInterpolationSideBase<EvalT,Traits,ScalarType>(*p,dl->side_layouts.at(sideSetName)));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructP0InterpolationEvaluator(
    const std::string& dof_name,
    const std::string& interpolationType,
    const FieldLocation loc,
    const FieldRankType rank,
    const Teuchos::RCP<IntrepidBasis>& basis) const
{
  Teuchos::RCP<Teuchos::ParameterList> p;
  p = Teuchos::rcp(new Teuchos::ParameterList("DOF Nodes to Cell Interpolation "+dof_name));

  // Input
  p->set<std::string>("BF Name", "BF");
  p->set<std::string>("Field Name", dof_name);
  p->set<std::string>("Weighted Measure Name", "Weights");
  p->set<FieldLocation>("Field Location", loc);
  p->set<FieldRankType>("Field Rank Type", rank);
  p->set<std::string>("Interpolation Type", interpolationType);
  p->set<Teuchos::RCP<IntrepidBasis>>("Intrepid2 Basis", basis);

  // Output
  p->set<std::string>("Field P0 Name", dof_name);

  return Teuchos::rcp(new PHAL::P0InterpolationBase<EvalT,Traits,ScalarType>(*p,dl));
}

template<typename EvalT, typename Traits, typename ScalarType>
Teuchos::RCP< PHX::Evaluator<Traits> >
EvaluatorUtilsImpl<EvalT,Traits,ScalarType>::constructP0InterpolationSideEvaluator(
    const std::string& sideSetName,
    const std::string& dof_name,
    const std::string& interpolationType,
    const FieldLocation loc,
    const FieldRankType rank,
    const Teuchos::RCP<IntrepidBasis>& basis) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! The layout structure for side set " << sideSetName << " was not found.\n");

  Teuchos::RCP<Teuchos::ParameterList> p;
  p = Teuchos::rcp(new Teuchos::ParameterList("DOF Side Nodes to Side Interpolation "+dof_name));

  // Input
  p->set<std::string>("BF Name", "BF "+sideSetName);
  p->set<std::string>("Field Name", dof_name);
  p->set<std::string>("Weighted Measure Name", "Weighted Measure "+sideSetName);
  p->set<FieldLocation>("Field Location", loc);
  p->set<FieldRankType>("Field Rank Type", rank);
  p->set<std::string>("Side Set Name", sideSetName);
  p->set<std::string>("Interpolation Type", interpolationType);
  p->set<Teuchos::RCP<IntrepidBasis>>("Intrepid2 Basis", basis);

  // Output
  p->set<std::string>("Field P0 Name", dof_name);

  return Teuchos::rcp(new PHAL::P0InterpolationBase<EvalT,Traits,ScalarType>(*p,dl->side_layouts.at(sideSetName)));
}

} // namespace Albany
