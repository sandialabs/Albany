//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_DataTypes.hpp"

#include "Intrepid2_HGRAD_LINE_Cn_FEM.hpp"

#include "PHAL_GatherSolution.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_ScatterResidual.hpp"
#include "PHAL_MapToPhysicalFrame.hpp"
#include "PHAL_MapToPhysicalFrameSide.hpp"
#include "PHAL_ComputeBasisFunctions.hpp"
#include "PHAL_ComputeBasisFunctionsSide.hpp"
#include "PHAL_DOFCellToSide.hpp"
#include "PHAL_DOFGradInterpolation.hpp"
#include "PHAL_DOFGradInterpolationSide.hpp"
#include "PHAL_DOFInterpolation.hpp"
#include "PHAL_DOFInterpolationSide.hpp"
#include "PHAL_DOFTensorInterpolation.hpp"
#include "PHAL_DOFTensorGradInterpolation.hpp"
#include "PHAL_DOFVecCellToSide.hpp"
#include "PHAL_DOFVecGradInterpolation.hpp"
#include "PHAL_DOFVecInterpolation.hpp"
#include "PHAL_DOFVecInterpolationSide.hpp"
#include "PHAL_NodesToCellInterpolation.hpp"
#include "PHAL_QuadPointsToCellInterpolation.hpp"
#include "PHAL_SideQuadPointsToSideInterpolation.hpp"


/********************  Problem Utils Class  ******************************/

template<typename EvalT, typename Traits>
Albany::EvaluatorUtils<EvalT,Traits>::EvaluatorUtils(
     Teuchos::RCP<Albany::Layouts> dl_) :
     dl(dl_)
{
}


template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherSolutionEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF)
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


template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherSolutionEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       int offsetToFirstDOF)
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


template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherSolutionEvaluator_withAcceleration(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF)
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

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherSolutionEvaluator_withAcceleration(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       Teuchos::ArrayRCP<std::string> dof_names_dot,
       Teuchos::ArrayRCP<std::string> dof_names_dotdot,
       int offsetToFirstDOF)
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


template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherSolutionEvaluator_noTransient(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF)
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

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherSolutionEvaluator_noTransient(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> dof_names,
       int offsetToFirstDOF)
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

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherScalarNodalParameter(
       const std::string& param_name,
       const std::string& field_name)
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

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructScatterResidualEvaluator(
       bool isVectorField,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF, std::string scatterName)
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

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructScatterResidualEvaluator(
       int tensorRank,
       Teuchos::ArrayRCP<std::string> resid_names,
       int offsetToFirstDOF, std::string scatterName)
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

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherCoordinateVectorEvaluator(std::string strCurrentDisp)
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

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructMapToPhysicalFrameEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature,
    const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis)

{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Map To Physical Frame"));

    // Input: X, Y at vertices
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<RCP <Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set< RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > >
        ("Intrepid2 Basis", intrepidBasis);

    // Output: X, Y at Quad Points (same name as input)

    return rcp(new PHAL::MapToPhysicalFrame<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructMapToPhysicalFrameSideEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature,
    const std::string& sideSetName)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("Map To Physical Frame Side"));

    // Input: X, Y at vertices
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<std::string>("Coordinate Side QP Vector Name", "Coord Vec " + sideSetName);
    p->set< RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name", sideSetName);

    // Output: X, Y at Quad Points (same name as input)
    return rcp(new PHAL::MapToPhysicalFrameSide<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructComputeBasisFunctionsEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis,
    const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubature)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > >("Cubature", cubature);

    p->set< RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > >
        ("Intrepid2 Basis", intrepidBasis);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<std::string>("Weights Name",          "Weights");
    p->set<std::string>("Jacobian Det Name",          "Jacobian Det");
    p->set<std::string>("Jacobian Name",          "Jacobian");
    p->set<std::string>("Jacobian Inv Name",          "Jacobian Inv");
    p->set<std::string>("BF Name",          "BF");
    p->set<std::string>("Weighted BF Name", "wBF");

    p->set<std::string>("Gradient BF Name",          "Grad BF");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");

    return rcp(new PHAL::ComputeBasisFunctions<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructComputeBasisFunctionsSideEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasisSide,
    const Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubatureSide,
    const std::string& sideSetName)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions Side"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<std::string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > >("Cubature Side", cubatureSide);
    p->set< RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > >("Intrepid Basis Side", intrepidBasisSide);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name",sideSetName);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<std::string>("Weighted Measure Name",     "Weighted Measure "+sideSetName);
    p->set<std::string>("Metric Determinant Name",   "Metric Determinant "+sideSetName);
    p->set<std::string>("BF Name",                   "BF "+sideSetName);
    p->set<std::string>("Gradient BF Name",          "Grad BF "+sideSetName);
    p->set<std::string>("Inverse Metric Name",       "Inv Metric "+sideSetName);

    return rcp(new PHAL::ComputeBasisFunctionsSide<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFCellToSideEvaluator(
       const std::string& cell_dof_name,
       const std::string& sideSetName,
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& side_dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Cell To Side"));

    // Input
    p->set<std::string>("Cell Variable Name", cell_dof_name);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name", sideSetName);

    // Output
    if (side_dof_name!="")
      p->set<std::string>("Side Variable Name", side_dof_name);
    else
      p->set<std::string>("Side Variable Name", cell_dof_name);

    return rcp(new PHAL::DOFCellToSide<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFInterpolationEvaluator_noDeriv(
       const std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation "+dof_name));
    // Input
    p->set<string>("Variable Name", dof_name);
    p->set<string>("BF Name", "BF");

    // Output (assumes same Name as input)

    return rcp(new PHAL::DOFInterpolation_noDeriv<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFGradInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF)
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

    return rcp(new PHAL::DOFGradInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFGradInterpolationSideEvaluator(
       const std::string& dof_name,
       const std::string& sideSetName)
{
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

    return rcp(new PHAL::DOFGradInterpolationSide<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFGradInterpolationEvaluator_noDeriv(
    const std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Grad Interpolation "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("Gradient BF Name", "Grad BF");

    // Output (assumes same Name as input)
    p->set<std::string>("Gradient Variable Name", dof_name+" Gradient");

    return rcp(new PHAL::DOFGradInterpolation_noDeriv<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF)
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

    return rcp(new PHAL::DOFInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFInterpolationSideEvaluator(
       const std::string& dof_name,
       const std::string& sideSetName)
{
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

    return rcp(new PHAL::DOFInterpolationSide<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFTensorInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF)
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

    return rcp(new PHAL::DOFTensorInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFTensorGradInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF)
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

    return rcp(new PHAL::DOFTensorGradInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFVecCellToSideEvaluator(
       const std::string& cell_dof_name,
       const std::string& sideSetName,
       const Teuchos::RCP<shards::CellTopology>& cellType,
       const std::string& side_dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Vec Cell To Side"));

    // Input
    p->set<std::string>("Cell Variable Name", cell_dof_name);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Side Set Name", sideSetName);

    // Output
    if (side_dof_name!="")
      p->set<std::string>("Side Variable Name", side_dof_name);
    else
      p->set<std::string>("Side Variable Name", cell_dof_name);

    return rcp(new PHAL::DOFVecCellToSide<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFVecGradInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF)
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

    return rcp(new PHAL::DOFVecGradInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFVecInterpolationEvaluator(
       const std::string& dof_name,
       int offsetToFirstDOF)
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

    return rcp(new PHAL::DOFVecInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFVecInterpolationSideEvaluator(
       const std::string& dof_name,
       const std::string& sideSetName)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Vec Interpolation Side "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("BF Name", "BF "+sideSetName);
    p->set<std::string>("Side Set Name",sideSetName);

    // Output (assumes same Name as input)

    return rcp(new PHAL::DOFVecInterpolationSide<EvalT, Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFVecInterpolationSideParamEvaluator(
       const std::string& dof_name,
       const std::string& sideSetName)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Vec Interpolation Side "+dof_name));
    // Input
    p->set<std::string>("Variable Name", dof_name);
    p->set<std::string>("BF Name", "BF "+sideSetName);
    p->set<std::string>("Side Set Name",sideSetName);

    // Output (assumes same Name as input)

    return rcp(new PHAL::DOFVecInterpolationSideParam<EvalT, Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructNodesToCellInterpolationEvaluator(
  const std::string& dof_name,
  bool isVectorField)
{
  Teuchos::RCP<Teuchos::ParameterList> p;
  p = Teuchos::rcp(new Teuchos::ParameterList("DOF Nodes to Cell Interpolation "+dof_name));

  // Input
  p->set<std::string>("BF Variable Name", "BF");
  p->set<std::string>("Field Node Name", dof_name);
  p->set<std::string>("Weighted Measure Name", "Weights");
  p->set<bool>("Is Vector Field", isVectorField);

  // Output
  p->set<std::string>("Field Cell Name", dof_name);

  return Teuchos::rcp(new PHAL::NodesToCellInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructQuadPointsToCellInterpolationEvaluator(
  const std::string& dof_name,
  bool isVectorField)
{
  Teuchos::RCP<Teuchos::ParameterList> p;
  p = Teuchos::rcp(new Teuchos::ParameterList("DOF QuadPoint to Cell Interpolation "+dof_name));

  // Input
  p->set<std::string>("Field QP Name", dof_name);
  p->set<std::string>("Weighted Measure Name", "Weights");
  p->set<bool>("Is Vector Field", isVectorField);

  // Output
  p->set<std::string>("Field Cell Name", dof_name);

  return Teuchos::rcp(new PHAL::QuadPointsToCellInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructSideQuadPointsToSideInterpolationEvaluator(
  const std::string& dof_name,
  const std::string& sideSetName,
  bool isVectorField)
{
  Teuchos::RCP<Teuchos::ParameterList> p;
  p = Teuchos::rcp(new Teuchos::ParameterList("DOF Side QuadPoint to Side Interpolation "+dof_name));

  // Input
  p->set<std::string>("Field QP Name", dof_name);
  p->set<std::string>("Weighted Measure Name", "Weighted Measure "+sideSetName);
  p->set<std::string>("Side Set Name", sideSetName);
  p->set<bool>("Is Vector Field", isVectorField);

  // Output
  p->set<std::string>("Field Side Name", dof_name);

  return Teuchos::rcp(new PHAL::SideQuadPointsToSideInterpolation<EvalT,Traits>(*p,dl));
}
