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

#include "LCM/LCM_FactoryTraits.hpp"
#include "LameProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_Utils.hpp"

Albany::LameProblem::
LameProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
            const Teuchos::RCP<ParamLib>& paramLib_,
            const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveSource(false)
{
 
  std::string& method = params->get("Name", "Library of Advanced Materials for Engineering (LAME) ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  dofNames.resize(neq);
  dofNames[0] = "X";
  if (neq>1) dofNames[1] = "Y";
  if (neq>2) dofNames[2] = "Z";

  // currently only support 3D analyses
  TEST_FOR_EXCEPTION(neq != 3,
                     Teuchos::Exceptions::InvalidParameter,
                     "\nOnly three-dimensional analyses are suppored when using the Library of Advanced Materials for Engineering (LAME)\n");
}

Albany::LameProblem::
~LameProblem()
{
}

void
Albany::LameProblem::
buildProblem(
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(meshSpecs, stateMgr);
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
Albany::LameProblem::constructEvaluators(
       const Albany::MeshSpecsStruct& meshSpecs,
       Albany::StateManager& stateMgr)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using std::map;
   using LCM::FactoryTraits;
   using PHAL::AlbanyTraits;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = this->getIntrepidBasis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numDim = cubature->getDimension();
   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();
   cout << "XXXX USING NODES FOR VERTICES" << endl;
   //   const int numVertices = cellType->getVertexCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;

   // Parser will build parameter list that determines the field
   // evaluators to build
   map<string, RCP<ParameterList> > evaluators_to_build;

   RCP<DataLayout> node_scalar = rcp(new MDALayout<Cell,Node>(worksetSize,numNodes));
   RCP<DataLayout> qp_scalar = rcp(new MDALayout<Cell,QuadPoint>(worksetSize,numQPts));

   RCP<DataLayout> node_vector = rcp(new MDALayout<Cell,Node,Dim>(worksetSize,numNodes,numDim));
   RCP<DataLayout> qp_vector = rcp(new MDALayout<Cell,QuadPoint,Dim>(worksetSize,numQPts,numDim));
   RCP<DataLayout> qp_tensor = rcp(new MDALayout<Cell,QuadPoint,Dim,Dim>(worksetSize,numQPts,numDim,numDim));

   RCP<DataLayout> vertices_vector = 
     rcp(new MDALayout<Cell,Vertex, Dim>(worksetSize,numVertices,numDim));
   // Basis functions, Basis function gradient
   RCP<DataLayout> node_qp_scalar =
     rcp(new MDALayout<Cell,Node,QuadPoint>(worksetSize,numNodes, numQPts));
   RCP<DataLayout> node_qp_vector =
     rcp(new MDALayout<Cell,Node,QuadPoint,Dim>(worksetSize,numNodes, numQPts,numDim));

   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));

  { // Gather Solution
   RCP< vector<string> > dof_names = rcp(new vector<string>(1));
     (*dof_names)[0] = "Displacement";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set<bool>("Vector Field", true);
    p->set< RCP<DataLayout> >("Data Layout", node_vector);

     RCP< vector<string> > dof_names_dotdot = rcp(new vector<string>(1));
       (*dof_names_dotdot)[0] = "Displacement_dotdot";

     p->set< RCP< vector<string> > >("Time Dependent Solution Names", dof_names_dotdot);

    evaluators_to_build["Gather Solution"] = p;
  }

  { // Gather Coordinate Vector
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity Gather Coordinate Vector"));
    int type = FactoryTraits<AlbanyTraits>::id_gather_coordinate_vector;
    p->set<int>                ("Type", type);

    // Output:: Coordindate Vector at vertices
    p->set< RCP<DataLayout> >  ("Coordinate Data Layout",  vertices_vector);
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    evaluators_to_build["Gather Coordinate Vector"] = p;
  }

  // UNUSED FOR NOW
  { // Map To Physical Frame: Interpolate X, Y to QuadPoints
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity Map To Physical Frame"));

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

  { // Compute Basis Functions
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity Compute Basis Functions"));

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

  { // DOFVec: Interpolate nodal Displacement values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity DOFVecInterpolation Displacement"));

    int type = FactoryTraits<AlbanyTraits>::id_dofvec_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("Node Vector Data Layout",      node_vector);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOFVec Displacement"] = p;
  }

  {
   // DOF: Interpolate nodal Displacement Dot  values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity DOFVecInterpolation Displacement Dot"));

    int type = FactoryTraits<AlbanyTraits>::id_dofvec_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "Displacement_dotdot");
    p->set< RCP<DataLayout> >("Node Vector Data Layout",      node_vector);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOFVec Displacement_dotdot"] = p;
  }

  { // DOFVecGrad: Interpolate nodal Displacement gradients to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Elasticity DOFVecInterpolation Displacement Grad"));

    int type = FactoryTraits<AlbanyTraits>::id_dofvec_grad_interpolation;
    p->set<int>   ("Type", type);
    // Input
    p->set<string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("Node Vector Data Layout",      node_vector);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    // Output
    p->set<string>("Gradient Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    evaluators_to_build["DOFVec Grad Displacement"] = p;
  }

  if (haveSource) { // Source
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       "Error!  Sources not implemented in Elasticity yet!");

    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_source;
    p->set<int>("Type", type);

    p->set<string>("Source Name", "Source");
    p->set<string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["Source"] = p;
  }

  { // Strain
    RCP<ParameterList> p = rcp(new ParameterList("Strain"));

    int type = LCM::FactoryTraits<AlbanyTraits>::id_strain;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Output
    p->set<string>("Strain Name", "Strain"); //qp_tensor also

    evaluators_to_build["Strain"] = p;
  }

  { // Deformation Gradient
    RCP<ParameterList> p = rcp(new ParameterList("DefGrad"));

    int type = FactoryTraits<AlbanyTraits>::id_defgrad;
    p->set<int>("Type", type);

    //Input
    // If true, compute determinate of deformation gradient at all integration points, then replace all of them with the simple average for the element.  This give a constant volumetric response.
    const bool avgJ = params->get("avgJ", false);
    p->set<bool>("avgJ Name", avgJ);
    // If true, compute determinate of deformation gradient at all integration points, then replace all of them with the volume average for the element (integrate J over volume of element, divide by total volume).  This give a constant volumetric response.
    const bool volavgJ = params->get("volavgJ", false);
    p->set<bool>("volavgJ Name", volavgJ);
    // Integration weights for each quadrature point
    p->set<string>("Weights Name","Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set<string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    //Output
    p->set<string>("DefGrad Name", "Deformation Gradient"); //qp_tensor also
    p->set<string>("DetDefGrad Name", "Determinant of Deformation Gradient"); 
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DefGrad"] = p;
  }

  { // LameStress
    RCP<ParameterList> p = rcp(new ParameterList("Stress"));

    int type = LCM::FactoryTraits<AlbanyTraits>::id_lame_stress;
    p->set<int>("Type", type);

    // Material properties that will be passed to LAME material model
    string lameMaterialModel = params->get<string>("Lame Material Model");
    p->set<string>("Lame Material Model", lameMaterialModel);
    Teuchos::ParameterList& lameMaterialParametersList = p->sublist("Lame Material Parameters");
    lameMaterialParametersList = params->sublist("Lame Material Parameters");

    // Input
    p->set<string>("Strain Name", "Strain");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    p->set<string>("DefGrad Name", "Deformation Gradient"); // qp_tensor also

    // Output
    p->set<string>("Stress Name", "Stress"); // qp_tensor also

    evaluators_to_build["Stress"] = p;

    // Declare state data that need to be saved
    // (register with state manager and create corresponding evaluator)

    // Stress and DefGrad are required at the element level for all LAME models
    evaluators_to_build["Save Stress"] =
      stateMgr.registerStateVariable("Stress",qp_tensor,
            dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"zero",true);
    evaluators_to_build["Save DefGrad"] =
      stateMgr.registerStateVariable("Deformation Gradient",qp_tensor,
            dummy, FactoryTraits<AlbanyTraits>::id_savestatefield,"identity", true);

    // A LAME material model may register additional state variables (type is always double)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    std::vector<std::string> lameMaterialModelStateVariableNames = LameUtils::getStateVariableNames(lameMaterialModel, lameMaterialParametersList);
    std::vector<double> lameMaterialModelStateVariableInitialValues = LameUtils::getStateVariableInitialValues(lameMaterialModel, lameMaterialParametersList);
    for(unsigned int i=0 ; i<lameMaterialModelStateVariableNames.size() ; ++i){
      evaluators_to_build["Save " + lameMaterialModelStateVariableNames[i]] =
        stateMgr.registerStateVariable(lameMaterialModelStateVariableNames[i],
                                       qp_scalar,
                                       dummy,
                                       FactoryTraits<AlbanyTraits>::id_savestatefield,
                                       doubleToInitString(lameMaterialModelStateVariableInitialValues[i], true));
    }
  }

  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    int type = LCM::FactoryTraits<AlbanyTraits>::id_elasticityresid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Stress Name", "Stress");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", qp_tensor);

    // \todo Is the required?
    p->set<string>("DefGrad Name", "Deformation Gradient"); //qp_tensor also

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    // extra input for time dependent term
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);
    p->set<string>("Time Dependent Variable Name", "Displacement_dotdot");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    //Output
    p->set<string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", node_vector);

    evaluators_to_build["Elasticity Resid"] = p;
  }

  { // Scatter Residual
   RCP< vector<string> > resid_names = rcp(new vector<string>(1));
     (*resid_names)[0] = "Displacement Residual";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_scatter_residual;
    p->set<int>("Type", type);
    p->set<bool>("Vector Field", true);
    p->set< RCP< vector<string> > >("Residual Names", resid_names);

    p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
    p->set< RCP<DataLayout> >("Data Layout", node_vector);

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

   // States to output every residual fill
   const Albany::StateManager::RegisteredStates& reg = stateMgr.getRegisteredStates();
   Albany::StateManager::RegisteredStates::const_iterator st = reg.begin();
   while (st != reg.end()) {
     PHX::Tag<AlbanyTraits::Residual::ScalarT> res_out_tag(st->first, dummy);
     fm->requireField<AlbanyTraits::Residual>(res_out_tag);
     st++;
   }
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::LameProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidLameProblemParams");

  validPL->set<string>("Lame Material Model", "", "The name of the LAME material model.");
  validPL->sublist("Lame Material Parameters", false, "");
  validPL->sublist("aveJ", false, "If true, the determinate of the deformation gradient for each integration point is replaced with the average value over all integration points in the element (produces constant volumetric response).");
  validPL->sublist("volaveJ", false, "If true, the determinate of the deformation gradient for each integration point is replaced with the volume-averaged value over all integration points in the element (produces constant volumetric response).");

  return validPL;
}

