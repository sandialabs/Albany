//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_DataTypes.hpp"

#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"

#include "PHAL_GatherSolution.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_GatherSHeight.hpp"
#include "PHAL_ScatterResidual.hpp"
#include "PHAL_MapToPhysicalFrame.hpp"
#include "PHAL_ComputeBasisFunctions.hpp"
#include "PHAL_DOFInterpolation.hpp"
#include "PHAL_DOFGradInterpolation.hpp"
#include "PHAL_DOFVecInterpolation.hpp"
#include "PHAL_DOFVecGradInterpolation.hpp"

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
    p->set< Teuchos::ArrayRCP<string> >("Solution Names", dof_names);

    p->set<bool>("Vector Field", isVectorField);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    p->set< Teuchos::ArrayRCP<string> >("Time Dependent Solution Names", dof_names_dot);
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
    p->set< Teuchos::ArrayRCP<string> >("Solution Names", dof_names);

    p->set<bool>("Vector Field", isVectorField);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<bool>("Disable Transient", true);

    return rcp(new PHAL::GatherSolution<EvalT,Traits>(*p,dl));
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
    p->set< Teuchos::ArrayRCP<string> >("Residual Names", resid_names);

    p->set<bool>("Vector Field", isVectorField);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<string>("Scatter Field Name", scatterName);

    return rcp(new PHAL::ScatterResidual<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherCoordinateVectorEvaluator()
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Coordinate Vector"));

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);
 
    // Output:: Coordindate Vector at vertices
    p->set<string>("Coordinate Vector Name", "Coord Vec");
 
    return rcp(new PHAL::GatherCoordinateVector<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherSHeightEvaluator()
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Surface Height"));

    // Output:: Surface Height Vector at vertices
    p->set<string>("Surface Height Name", "Surface Height");

    return rcp(new PHAL::GatherSHeight<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructMapToPhysicalFrameEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Map To Physical Frame"));
 
    // Input: X, Y at vertices
    p->set<string>("Coordinate Vector Name", "Coord Vec");
 
    p->set<RCP <Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
 
    // Output: X, Y at Quad Points (same name as input)
 
    return rcp(new PHAL::MapToPhysicalFrame<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructComputeBasisFunctionsEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis,
    const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
 
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);
 
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Weights Name",          "Weights");
    p->set<string>("Jacobian Det Name",          "Jacobian Det");
    p->set<string>("BF Name",          "BF");
    p->set<string>("Weighted BF Name", "wBF");
 
    p->set<string>("Gradient BF Name",          "Grad BF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");

    return rcp(new PHAL::ComputeBasisFunctions<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFInterpolationEvaluator(
       std::string& dof_name)
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

    return rcp(new PHAL::DOFInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFGradInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Grad Interpolation "+dof_name));
    // Input
    p->set<string>("Variable Name", dof_name);

    p->set<string>("Gradient BF Name", "Grad BF");

    // Output (assumes same Name as input)
    p->set<string>("Gradient Variable Name", dof_name+" Gradient");

    return rcp(new PHAL::DOFGradInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFGradInterpolationEvaluator_noDeriv(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Grad Interpolation "+dof_name));
    // Input
    p->set<string>("Variable Name", dof_name);

    p->set<string>("Gradient BF Name", "Grad BF");

    // Output (assumes same Name as input)
    p->set<string>("Gradient Variable Name", dof_name+" Gradient");

    return rcp(new PHAL::DOFGradInterpolation_noDeriv<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFVecInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("DOFVec Interpolation "+dof_name));
    // Input
    p->set<string>("Variable Name", dof_name);

    p->set<string>("BF Name", "BF");

    // Output (assumes same Name as input)

    return rcp(new PHAL::DOFVecInterpolation<EvalT,Traits>(*p,dl));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFVecGradInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("DOFVecGrad Interpolation "+dof_name));
    // Input
    p->set<string>("Variable Name", dof_name);

    p->set<string>("Gradient BF Name", "Grad BF");

    // Output (assumes same Name as input)
    p->set<string>("Gradient Variable Name", dof_name+" Gradient");

    return rcp(new PHAL::DOFVecGradInterpolation<EvalT,Traits>(*p,dl));
}
