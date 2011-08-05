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


#include "Albany_Helmholtz2DProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"
#include "Albany_Utils.hpp"

Albany::Helmholtz2DProblem::
Helmholtz2DProblem(
                         const Teuchos::RCP<Teuchos::ParameterList>& params_,
                         const Teuchos::RCP<ParamLib>& paramLib_) :
  Albany::AbstractProblem(params_, paramLib_, 2)
{

  std::string& method = params->get("Name", "Helmholtz 2D Problem");
  *out << "Problem Name = " << method << std::endl;
  
  ksqr = params->get<double>("Ksqr",1.0);

  haveSource =  params->isSublist("Source Functions");

  dofNames.resize(neq);
  dofNames[0] = "U";
  dofNames[1] = "V";
}

Albany::Helmholtz2DProblem::
~Helmholtz2DProblem()
{
}

void
Albany::Helmholtz2DProblem::
buildProblem(
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(meshSpecs);
  constructDirichletEvaluators(meshSpecs.nsNames);

  // Build response functions
  Teuchos::ParameterList& responseList = params->sublist("Response Functions");
  int num_responses = responseList.get("Number", 0);
  responses.resize(num_responses);
  for (int i=0; i<num_responses; i++) {
     std::string name = responseList.get(Albany::strint("Response",i), "??");

     if (name == "Solution Average")
       responses[i] = Teuchos::rcp(new SolutionAverageResponseFunction());

     else if (name == "Solution Two Norm")
       responses[i] = Teuchos::rcp(new SolutionTwoNormResponseFunction());

     else {
       TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                          std::endl <<
                          "Error!  Unknown response function " << name <<
                          "!" << std::endl << "Supplied parameter list is " <<
                          std::endl << responseList);
     }

  }
}


void
Albany::Helmholtz2DProblem::constructEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using std::map;
   using PHAL::FactoryTraits;
   using PHAL::AlbanyTraits;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology(&meshSpecs.ctd)); 
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = this->getIntrepidBasis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;

   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numDim = cubature->getDimension();
   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getVertexCount();

   *out << "Field Dimensions: Workset=" << worksetSize
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;

   // Parser will build parameter list that determines the field
   // evaluators to build
   map<string, RCP<ParameterList> > evaluators_to_build;

   // "node" refers to nodal degrees of freedom (not necessarily the same as cell vertices)
   RCP<DataLayout> node_scalar = rcp(new MDALayout<Cell,Node>(worksetSize,numNodes));
   RCP<DataLayout> node_vector = rcp(new MDALayout<Cell,Node,Dim>(worksetSize,numNodes,numDim));
   RCP<DataLayout> qp_scalar = rcp(new MDALayout<Cell,QuadPoint>(worksetSize,numQPts));
   RCP<DataLayout> qp_vector = rcp(new MDALayout<Cell,QuadPoint,Dim>(worksetSize,numQPts,numDim));
   //RCP<DataLayout> vertices_scalar = rcp(new MDALayout<Cell,Vertex>(worksetSize,numVertices));
   RCP<DataLayout> vertices_vector = rcp(new MDALayout<Cell,Vertex, Dim>(worksetSize,numVertices,numDim));

   // Basis functions, Basis function gradient
   RCP<DataLayout> node_qp_scalar = rcp(new MDALayout<Cell,Node,QuadPoint>(worksetSize,numNodes, numQPts));
   RCP<DataLayout> node_qp_vector = rcp(new MDALayout<Cell,Node,QuadPoint,Dim>(worksetSize,numNodes, numQPts,numDim));

   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));

#ifdef GATHER_BOTH_U_AND_V_TOGETHER
 // This version grabs both U and V together
  { // Gather Solution
   RCP< vector<string> > dof_names = rcp(new vector<string>(neq));
     (*dof_names)[0] = "U";
     (*dof_names)[1] = "V";

    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Gather Solution"));
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);

    // Helmholtz solve does not have transient terms
    p->set<bool>("Disable Transient", true);

    evaluators_to_build["Gather Solution"] = p;
  }
#else
  { // Gather Solution
   RCP< vector<string> > dof_names = rcp(new vector<string>(1));
     (*dof_names)[0] = "U";

    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Gather Solution"));
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);
    p->set<int>("Offset of First DOF", 0);

    // Helmholtz solve does not have transient terms
    p->set<bool>("Disable Transient", true);

    evaluators_to_build["Gather U Solution"] = p;
  }
  { // Gather Solution
   RCP< vector<string> > dof_names = rcp(new vector<string>(1));
     (*dof_names)[0] = "V";

    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Gather Solution"));
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);
    p->set<int>("Offset of First DOF", 1);

    // Helmholtz solve does not have transient terms
    p->set<bool>("Disable Transient", true);

    evaluators_to_build["Gather V Solution"] = p;
  }
#endif

  { // Gather Coordinate Vector
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Gather Coordinate Vector"));
    int type = FactoryTraits<AlbanyTraits>::id_gather_coordinate_vector;
    p->set<int>                ("Type", type);
    p->set<bool>("Periodic BC", false);

    p->set< string >("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >  ("Coordinate Data Layout",  vertices_vector);
    evaluators_to_build["Gather Coordinate Vector"] = p;
  }

  { // Compute Basis Functions
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Compute Basis Functions"));

    int type = FactoryTraits<AlbanyTraits>::id_compute_basis_functions;
    p->set<int>   ("Type", type);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", vertices_vector);
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
        ("Intrepid Basis", intrepidBasis);

    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>("Weights Name",          "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set<string>("BF Name",          "BF");
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    p->set<string>("Gradient BF Name",          "Grad BF");
    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    evaluators_to_build["Compute Basis Functions"] = p;
  }

  { // DOF: Interpolate nodal U values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz DOFInterpolation U"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "U");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF U"] = p;
  }

  { // DOF: Interpolate nodal V values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz DOFInterpolation V"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "V");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF V"] = p;
  }

  { // DOFGrad: Interpolate nodal U to Grad-U at quad points
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz DOFGradInterpolation U"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_grad_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "U");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);
 
    // Output
    p->set<string>("Gradient Variable Name", "GradU");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOF Grad U"] = p;
  }

  { // DOFGrad: Interpolate nodal V to Grad-V at quad points
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz DOFGradInterpolation V"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_grad_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "V");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);
 
    // Output
    p->set<string>("Gradient Variable Name", "GradV");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOF Grad V"] = p;
  }

  { // Map To Physical Frame: Interpolate X, Y to QuadPoints
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Map To Physical Frame"));

    int type = FactoryTraits<AlbanyTraits>::id_map_to_physical_frame;
    p->set<int>   ("Type", type);

    // Input: X, Y at vertices
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Coordinate Data Layout", vertices_vector);

    p->set<RCP <Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
 
    // Output: X, Y at Quad Points (same name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["Map To Physical Frame"] = p;
  }

  if (haveSource) { // Source on U (Real) equation
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Source U"));

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<string>("Pressure Source Name", "U GaussMonotone");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");


    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Source U"] = p;
  }

  if (haveSource) { // Source on V (Imag) equation
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Source V"));

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<string>("Pressure Source Name", "V GaussMonotone");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");


    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Source V"] = p;
  }

  { // Helmholtz Resid
    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_helmholtzresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    p->set<string>("U Variable Name", "U");
    p->set<string>("V Variable Name", "V");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    p->set<string>("U Gradient Variable Name", "GradU");
    p->set<string>("V Gradient Variable Name", "GradV");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<bool>("Have Source", haveSource);
    p->set<string>("U Pressure Source Name", "U GaussMonotone");
    p->set<string>("V Pressure Source Name", "V GaussMonotone");

    p->set<double>("Ksqr", ksqr);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<string>("U Residual Name", "U Residual");
    p->set<string>("V Residual Name", "V Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", node_scalar);

    evaluators_to_build["Helmholtz Resid"] = p;
  }


  { // Scatter Residual
   RCP< vector<string> > resid_names = rcp(new vector<string>(neq));
     (*resid_names)[0] = "U Residual";
     (*resid_names)[1] = "V Residual";

    RCP<ParameterList> p = rcp(new ParameterList("Helmholtz Scatter Residual"));
    int type = FactoryTraits<AlbanyTraits>::id_scatter_residual;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Residual Names", resid_names);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);

    evaluators_to_build["Scatter Residual"] = p;
  }

   // Build Field Evaluators for each evaluation type
   PHX::EvaluatorFactory<AlbanyTraits,FactoryTraits<AlbanyTraits> > factory;
   RCP< vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > >
     evaluators;
   evaluators = factory.buildEvaluators(evaluators_to_build);

   // Create a FieldManager
   fm = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

   // Register all Evaluators
   PHX::registerEvaluators(evaluators, *fm);

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::Residual>(res_tag);
   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::Tangent>(tan_tag);
   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter", dummy);
   fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::Helmholtz2DProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHelmhotz2DProblemParams");
  validPL->set<double>("Left BC", 0.0, "Value of Left BC [required]");
  validPL->set<double>("Right BC", 0.0, "Value to Right BC [required]");
  validPL->set<double>("Top BC", 0.0, "Value of Top BC [required]");
  validPL->set<double>("Bottom BC", 0.0, "Value to Bottom BC [required]");
  validPL->set<double>("Ksqr", 1.0, "Value of wavelength-squared [required]");

  return validPL;
}
