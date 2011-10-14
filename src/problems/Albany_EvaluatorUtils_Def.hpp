/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "Albany_EvaluatorUtils.hpp"
#include "Albany_DataTypes.hpp"

#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"

#include "PHAL_GatherSolution.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
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
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    p->set<bool>("Vector Field", isVectorField);
    if (isVectorField) p->set< RCP<DataLayout> >("Data Layout", dl->node_vector);
    else               p->set< RCP<DataLayout> >("Data Layout", dl->node_scalar);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);

    p->set< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names", dof_names_dot);
    return rcp(new PHAL::GatherSolution<EvalT,Traits>(*p));
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
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Solution"));
    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    p->set<bool>("Vector Field", isVectorField);
    if (isVectorField) p->set< RCP<DataLayout> >("Data Layout", dl->node_vector);
    else               p->set< RCP<DataLayout> >("Data Layout", dl->node_scalar);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<bool>("Disable Transient", true);

    return rcp(new PHAL::GatherSolution<EvalT,Traits>(*p));
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
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("Scatter Residual"));
    p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);

    p->set<bool>("Vector Field", isVectorField);
    if (isVectorField) p->set< RCP<DataLayout> >("Data Layout", dl->node_vector);
    else               p->set< RCP<DataLayout> >("Data Layout", dl->node_scalar);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dl->dummy);
    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<string>("Scatter Field Name", scatterName);

    return rcp(new PHAL::ScatterResidual<EvalT,Traits>(*p));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructGatherCoordinateVectorEvaluator()
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("Gather Coordinate Vector"));

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);
 
    // Output:: Coordindate Vector at vertices
    p->set< RCP<DataLayout> >  ("Coordinate Data Layout",  dl->vertices_vector);
    p->set< string >("Coordinate Vector Name", "Coord Vec");
 
    return rcp(new PHAL::GatherCoordinateVector<EvalT,Traits>(*p));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructMapToPhysicalFrameEvaluator(
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("Map To Physical Frame"));
 
    // Input: X, Y at vertices
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
 
    p->set<RCP <Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
 
    // Output: X, Y at Quad Points (same name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);
 
    return rcp(new PHAL::MapToPhysicalFrame<EvalT,Traits>(*p));
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
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", dl->vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
 
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);
 
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
     p->set<string>("Weights Name",          "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set<string>("BF Name",          "BF");
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
 
    p->set<string>("Gradient BF Name",          "Grad BF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    return rcp(new PHAL::ComputeBasisFunctions<EvalT,Traits>(*p));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation "+dof_name));
    // Input
    p->set<string>("Variable Name", dof_name);
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    return rcp(new PHAL::DOFInterpolation<EvalT,Traits>(*p));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFGradInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOF Interpolation "+dof_name));
    // Input
    p->set<string>("Variable Name", dof_name);
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    // Output (assumes same Name as input)
    p->set<string>("Gradient Variable Name", dof_name+" Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    return rcp(new PHAL::DOFGradInterpolation<EvalT,Traits>(*p));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFVecInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOFVec Interpolation "+dof_name));
    // Input
    p->set<string>("Variable Name", dof_name);
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    return rcp(new PHAL::DOFVecInterpolation<EvalT,Traits>(*p));
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
Albany::EvaluatorUtils<EvalT,Traits>::constructDOFVecGradInterpolationEvaluator(
       std::string& dof_name)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using PHX::DataLayout;
    using Teuchos::ParameterList;

    RCP<ParameterList> p = rcp(new ParameterList("DOFVecGrad Interpolation "+dof_name));
    // Input
    p->set<string>("Variable Name", dof_name);
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    // Output (assumes same Name as input)
    p->set<string>("Gradient Variable Name", dof_name+" Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    return rcp(new PHAL::DOFVecGradInterpolation<EvalT,Traits>(*p));
}
