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


#include "QCAD_SchrodingerProblem.hpp"
#include "QCAD_MaterialDatabase.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"

//#include "Intrepid_HGRAD_LINE_C1_FEM.hpp"
//#include "Intrepid_HGRAD_QUAD_C1_FEM.hpp"
//#include "Intrepid_HGRAD_TRI_C1_FEM.hpp"
//#include "Intrepid_HGRAD_HEX_C1_FEM.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"

#include "Albany_Utils.hpp"


QCAD::SchrodingerProblem::
SchrodingerProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
		    const Teuchos::RCP<ParamLib>& paramLib_,
		    const int numDim_,
		    const Teuchos::RCP<const Epetra_Comm>& comm_) :
  Albany::AbstractProblem(params_, paramLib_, 1),
  comm(comm_),
  havePotential(false), haveMaterial(false),
  numDim(numDim_)
{
  havePotential = params->isSublist("Potential");
  haveMaterial = params->isSublist("Material");

  //Note: can't use get("ParamName" , <default>) form b/c non-const
  energy_unit_in_eV = 1e-3; //default meV
  if(params->isType<double>("EnergyUnitInElectronVolts"))
    energy_unit_in_eV = params->get<double>("EnergyUnitInElectronVolts");
  std::cout << "Energy unit = " << energy_unit_in_eV << " eV" << endl;

  length_unit_in_m = 1e-9; //default to nm
  if(params->isType<double>("LengthUnitInMeters"))
    length_unit_in_m = params->get<double>("LengthUnitInMeters");
  std::cout << "Length unit = " << length_unit_in_m << " meters" << endl;

  mtrlDbFilename = "materials.xml";
  if(params->isType<string>("MaterialDB Filename"))
    mtrlDbFilename = params->get<string>("MaterialDB Filename");


  potentialStateName = "V"; //default name for potential at QPs field
  //nEigenvectorsToOuputAsStates = 0;
  bOnlySolveInQuantumBlocks = false;

  //Poisson coupling
  if(params->isSublist("Poisson Coupling")) {
    Teuchos::ParameterList& cList = params->sublist("Poisson Coupling");
    if(cList.isType<bool>("Only solve in quantum blocks"))
      bOnlySolveInQuantumBlocks = cList.get<bool>("Only solve in quantum blocks");
    if(cList.isType<string>("Potential State Name"))
    potentialStateName = cList.get<string>("Potential State Name");

    //if(cList.isType<int>("Save Eigenvectors as States"))
    //  nEigenvectorsToOuputAsStates = cList.get<int>("Save Eigenvectors as States");
  }

  //Check LOCA params to see if eigenvectors will be output to states.
  //Teuchos::ParameterList& locaStepperList = params->sublist("LOCA").sublist("Stepper");
  //if( locaStepperList.get("Compute Eigenvalues", false) > 0) {
  //  int nSave = locaStepperList.sublist("Eigensolver").get("Save Eigenvectors",0);
  //  int nSaveAsStates = locaStepperList.sublist("Eigensolver").get("Save Eigenvectors as States", 0);
  //  nEigenvectorsToOuputAsStates = (nSave < nSaveAsStates)? nSave : nSaveAsStates;
  //}

  TEST_FOR_EXCEPTION(params->isSublist("Source Functions"), Teuchos::Exceptions::InvalidParameter,
		     "\nError! Schrodinger problem does not parse Source Functions sublist\n" 
                     << "\tjust Potential sublist " << std::endl);

  // neq=1 set in AbstractProblem constructor
  dofNames.resize(neq);
  dofNames[0] = "psi";
}

QCAD::SchrodingerProblem::
~SchrodingerProblem()
{
}

void
QCAD::SchrodingerProblem::
buildProblem(
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(meshSpecs, stateMgr, responses);
  constructDirichletEvaluators(meshSpecs.nsNames);
}


void
QCAD::SchrodingerProblem::constructEvaluators(
       const Albany::MeshSpecsStruct& meshSpecs,
       Albany::StateManager& stateMgr,
       std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
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

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = this->getIntrepidBasis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

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
   RCP<DataLayout> cell_scalar = rcp(new MDALayout<Cell,QuadPoint>(worksetSize,1));

   RCP<DataLayout> node_vector = rcp(new MDALayout<Cell,Node,Dim>(worksetSize,numNodes,numDim));
   RCP<DataLayout> qp_vector = rcp(new MDALayout<Cell,QuadPoint,Dim>(worksetSize,numQPts,numDim));

   RCP<DataLayout> vertices_vector = 
     rcp(new MDALayout<Cell,Vertex, Dim>(worksetSize,numVertices,numDim));
   // Basis functions, Basis function gradient
   RCP<DataLayout> node_qp_scalar =
     rcp(new MDALayout<Cell,Node,QuadPoint>(worksetSize,numNodes, numQPts));
   RCP<DataLayout> node_qp_vector =
     rcp(new MDALayout<Cell,Node,QuadPoint,Dim>(worksetSize,numNodes, numQPts,numDim));

   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));

   // Create Material Database
   RCP<QCAD::MaterialDatabase> materialDB = rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));

  { // Gather Solution
   RCP< vector<string> > dof_names = rcp(new vector<string>(neq));
     (*dof_names)[0] = "psi";

    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_solution;
    p->set<int>("Type", type);
    p->set< RCP< vector<string> > >("Solution Names", dof_names);
    p->set< RCP<DataLayout> >("Data Layout", node_scalar);

   RCP< vector<string> > dof_names_dot = rcp(new vector<string>(neq));
     (*dof_names_dot)[0] = "psi_dot";

   p->set< RCP< vector<string> > >("Time Dependent Solution Names", dof_names_dot);

    evaluators_to_build["Gather Solution"] = p;
  }

  { // Gather Coordinate Vector
    RCP<ParameterList> p = rcp(new ParameterList("Schrodinger Gather Coordinate Vector"));
    int type = FactoryTraits<AlbanyTraits>::id_gather_coordinate_vector;
    p->set<int>                ("Type", type);
    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);

    // Output:: Coordindate Vector at vertices
    p->set< RCP<DataLayout> >  ("Coordinate Data Layout",  vertices_vector);
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    evaluators_to_build["Gather Coordinate Vector"] = p;
  }

  { // Map To Physical Frame: Interpolate X, Y to QuadPoints
    RCP<ParameterList> p = rcp(new ParameterList("Schrodinger 1D Map To Physical Frame"));

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
    RCP<ParameterList> p = rcp(new ParameterList("Schrodinger Compute Basis Functions"));

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

  { // DOF: Interpolate nodal Wavefunction values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Schrodinger DOFInterpolation Wavefunction"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "psi");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF Wavefunction"] = p;
  }

  {
   // DOF: Interpolate nodal wavefunction Dot values to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Schrodinger DOFInterpolation Wavefunction Dot"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "psi_dot");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);

    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    evaluators_to_build["DOF Wavefunction_dot"] = p;
  }

  { // DOF: Interpolate nodal wavefunction gradients to quad points
    RCP<ParameterList> p = rcp(new ParameterList("Schrodinger DOFInterpolation Wavefunction Grad"));

    int type = FactoryTraits<AlbanyTraits>::id_dof_grad_interpolation;
    p->set<int>   ("Type", type);

    // Input
    p->set<string>("Variable Name", "psi");
    p->set< RCP<DataLayout> >("Node Data Layout",      node_scalar);

    p->set<string>("Gradient BF Name", "Grad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    // Output
    p->set<string>("Gradient Variable Name", "Grad psi");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    evaluators_to_build["DOF Grad Wavefunction"] = p;
  }

  if (havePotential) { // Potential energy
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_schrodinger_potential;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "psi");
    p->set<string>("QP Potential Name", potentialStateName);
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Potential");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Global Problem Parameters
    p->set<double>("Energy unit in eV", energy_unit_in_eV);
    p->set<double>("Length unit in m", length_unit_in_m);

    evaluators_to_build["Potential Energy"] = p;

    //DEBUG: Output potential to check that it is imported properly; but 
    // this can be done via response evaluators now, so this code is no longer needed
    /*std::cout << "DEBUG: potential from state name " << potentialStateName << std::endl;
    int issf = FactoryTraits<AlbanyTraits>::id_savestatefield;
    evaluators_to_build["Save Potential"] =
      stateMgr.registerStateVariable(potentialStateName, qp_scalar, dummy, issf);
    */
  }

  { // Wavefunction (psi) Resid
    RCP<ParameterList> p = rcp(new ParameterList("Wavefunction Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_schrodinger_resid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", node_qp_scalar);
    p->set<string>("QP Variable Name", "psi");
    p->set<string>("QP Time Derivative Variable Name", "psi_dot");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");

    p->set<bool>("Have Potential", havePotential);
    p->set<bool>("Have Material", haveMaterial);
    p->set<string>("Potential Name", potentialStateName); // was "V"
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Grad psi");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", node_qp_vector);

    //Output
    p->set<string>("Residual Name", "psi Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", node_scalar);

    if(haveMaterial) {
      Teuchos::ParameterList& paramList = params->sublist("Material");
      p->set<Teuchos::ParameterList*>("Material Parameter List", &paramList);
    }

    //Global Problem Parameters
    p->set<double>("Energy unit in eV", energy_unit_in_eV);
    p->set<double>("Length unit in m", length_unit_in_m);
    p->set<bool>("Only solve in quantum blocks", bOnlySolveInQuantumBlocks);
    p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    evaluators_to_build["psi Resid"] = p;
  }

  { // Scatter Residual
   RCP< vector<string> > resid_names = rcp(new vector<string>(neq));
     (*resid_names)[0] = "psi Residual";

    RCP<ParameterList> p = rcp(new ParameterList);
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

   //! Construct Responses
   Teuchos::ParameterList& responseList = params->sublist("Response Functions");
   constructResponses(responses, responseList, evaluators_to_build, stateMgr,
		      qp_scalar, qp_vector, cell_scalar, dummy);

}


void
QCAD::SchrodingerProblem::constructResponses(
  std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
  Teuchos::ParameterList& responseList, 
  std::map<string, Teuchos::RCP<Teuchos::ParameterList> >& evaluators_to_build, 
  Albany::StateManager& stateMgr,
  Teuchos::RCP<PHX::DataLayout> qp_scalar, Teuchos::RCP<PHX::DataLayout> qp_vector,
  Teuchos::RCP<PHX::DataLayout> cell_scalar, Teuchos::RCP<PHX::DataLayout> dummy)
{
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using std::string;
  using PHAL::FactoryTraits;
  using PHAL::AlbanyTraits;


   // Parameters for Response Evaluators
   //  Iterate through list of responses (from input xml file).  For each, create a response
   //  function and possibly a parameter list to construct a response evaluator.
   int num_responses = responseList.get("Number", 0);
   responses.resize(num_responses);

   std::map<string, RCP<ParameterList> > response_evaluators_to_build;
   std::vector<string> responseIDs_to_require;

   for (int i=0; i<num_responses; i++) 
   {
     std::string responseID = Albany::strint("Response",i);
     std::string name = responseList.get(responseID, "??");

     RCP<ParameterList> p;

     if( getStdResponseFn(name, i, responseList, responses, stateMgr,
			  qp_scalar, qp_vector, cell_scalar, dummy, p) ) {
       if(p != Teuchos::null) {
	 response_evaluators_to_build[responseID] = p;
	 responseIDs_to_require.push_back(responseID);
       }
     }

     else if (name == "Saddle Value")
     { 
       int type = FactoryTraits<AlbanyTraits>::id_qcad_response_saddlevalue;

       std::string responseParamsID = Albany::strint("ResponseParams",i);              
       ParameterList& responseParams = responseList.sublist(responseParamsID);
       RCP<ParameterList> p = rcp(new ParameterList);
       
       RCP<QCAD::SaddleValueResponseFunction> 
	 svResponse = Teuchos::rcp(new QCAD::SaddleValueResponseFunction(
					     numDim, responseParams)); 
       responses[i] = svResponse;
       
       p->set<string>("Response ID", responseID);
       p->set<int>   ("Response Index", i);
       p->set< Teuchos::RCP<QCAD::SaddleValueResponseFunction> >
	 ("Response Function", svResponse);
       p->set<Teuchos::ParameterList*>("Parameter List", &responseParams);
       p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);
       
       p->set<int>("Type", type);
       p->set<string>("Coordinate Vector Name", "Coord Vec");
       p->set<string>("Weights Name",   "Weights");
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
       p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);

       response_evaluators_to_build[responseID] = p;
       responseIDs_to_require.push_back(responseID);
     }

     else {
       TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
	  std::endl << "Error!  Unknown response function " << name <<
           "!" << std::endl << "Supplied parameter list is " <<
           std::endl << responseList);
     }
   } // end of loop over responses

   //! Create field manager for responses
   createResponseFieldManager(response_evaluators_to_build, 
			      evaluators_to_build, responseIDs_to_require, dummy);
}


void
QCAD::SchrodingerProblem::createResponseFieldManager(
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >& response_evaluators_to_build,
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >& evaluators_to_build,
    const std::vector<std::string>& responseIDs_to_require, Teuchos::RCP<PHX::DataLayout> dummy)
{
  using Teuchos::RCP;
  using std::string;
  using PHAL::AlbanyTraits;
  using PHAL::FactoryTraits;

  // Build Response Evaluators for each evaluation type
  PHX::EvaluatorFactory<AlbanyTraits,FactoryTraits<AlbanyTraits> > factory;
  RCP< std::vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > >
    response_evaluators;
   
  response_evaluators_to_build.insert(evaluators_to_build.begin(), evaluators_to_build.end());
  response_evaluators = factory.buildEvaluators(response_evaluators_to_build);

  // Create a Response FieldManager
  rfm = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

  // Register all Evaluators
  PHX::registerEvaluators(response_evaluators, *rfm);

  // Set required fields: ( Response<i>, dummy ), for responses 
  //  evaluated by the response evaluators
  std::vector<string>::const_iterator it;
  for (it = responseIDs_to_require.begin(); it != responseIDs_to_require.end(); it++)
  {
    const string& responseID = *it;

    PHX::Tag<AlbanyTraits::Residual::ScalarT> res_response_tag(responseID, dummy);
    rfm->requireField<AlbanyTraits::Residual>(res_response_tag);
    PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_response_tag(responseID, dummy);
    rfm->requireField<AlbanyTraits::Jacobian>(jac_response_tag);
    PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_response_tag(responseID, dummy);
    rfm->requireField<AlbanyTraits::Tangent>(tan_response_tag);
    PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_response_tag(responseID, dummy);
    rfm->requireField<AlbanyTraits::SGResidual>(sgres_response_tag);
    PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_response_tag(responseID, dummy);
    rfm->requireField<AlbanyTraits::SGJacobian>(sgjac_response_tag);
    PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_response_tag(responseID, dummy);
    rfm->requireField<AlbanyTraits::MPResidual>(mpres_response_tag);
    PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_response_tag(responseID, dummy);
    rfm->requireField<AlbanyTraits::MPJacobian>(mpjac_response_tag);
  }
}


Teuchos::RCP<Teuchos::ParameterList>
QCAD::SchrodingerProblem::setupResponseFnForEvaluator(
  Teuchos::ParameterList& responseList, int responseNumber,
  std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
  Teuchos::RCP<PHX::DataLayout> dummy)
{
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using std::string;

  std::string responseID = Albany::strint("Response",responseNumber);       
  std::string responseParamsID = Albany::strint("ResponseParams",responseNumber);       
  TEST_FOR_EXCEPTION(!responseList.isSublist(responseParamsID), 
		     Teuchos::Exceptions::InvalidParameter,
		     std::endl << Albany::strint("Response",responseNumber) <<
		     " requires a parameter list" << std::endl);

  ParameterList& responseParams = responseList.sublist(responseParamsID);
  RCP<ParameterList> p = rcp(new ParameterList);

  RCP<Albany::EvaluatedResponseFunction> 
    evResponse = Teuchos::rcp(new Albany::EvaluatedResponseFunction());
  responses[responseNumber] = evResponse;

  // Common parameters to all response evaluators
  p->set<string>("Response ID", responseID);
  p->set<int>   ("Response Index", responseNumber);
  p->set< RCP<Albany::EvaluatedResponseFunction> >("Response Function", evResponse);
  p->set<ParameterList*>("Parameter List", &responseParams);
  p->set< RCP<DataLayout> >("Dummy Data Layout", dummy);

  return p;
}


// - Returns true if responseName was recognized and response function constructed.
// - If p is non-Teuchos::null upon exit, then an evaluator should be build using
//   p as the parameter list. 
bool
QCAD::SchrodingerProblem::getStdResponseFn(
    std::string responseName, int responseIndex,
    Teuchos::ParameterList& responseList,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
    Albany::StateManager& stateMgr,    
    Teuchos::RCP<PHX::DataLayout> qp_scalar, Teuchos::RCP<PHX::DataLayout> qp_vector,
    Teuchos::RCP<PHX::DataLayout> cell_scalar, Teuchos::RCP<PHX::DataLayout> dummy,
    Teuchos::RCP<Teuchos::ParameterList>& p)
{
  using std::string;
  using Teuchos::RCP;
  using PHX::DataLayout;
  using PHAL::FactoryTraits;
  using PHAL::AlbanyTraits;


  p = Teuchos::null;

  if (responseName == "Field Integral") 
  {
    int type = FactoryTraits<AlbanyTraits>::id_qcad_response_fieldintegral;

    p = setupResponseFnForEvaluator(responseList, responseIndex, responses, dummy);
    p->set<int>("Type", type);
    p->set<string>("Weights Name",   "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);
    p->set<double>("Length unit in m", length_unit_in_m);
    return true;
  }

  else if (responseName == "Field Value") 
  { 
    int type = FactoryTraits<AlbanyTraits>::id_qcad_response_fieldvalue;

    p = setupResponseFnForEvaluator(responseList, responseIndex, responses, dummy);
    p->set<int>("Type", type);
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set<string>("Weights Name",   "Weights");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);
    return true;
  }

  else if (responseName == "Save Field")
  { 
    int type = FactoryTraits<AlbanyTraits>::id_qcad_response_savefield;
       
    p = setupResponseFnForEvaluator(responseList, responseIndex, responses, dummy);
    p->set<int>("Type", type);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", qp_vector);
    p->set< RCP<DataLayout> >("Cell Scalar Data Layout", cell_scalar);
    p->set< Albany::StateManager* >("State Manager Ptr", &stateMgr );
    p->set<string>("Weights Name",   "Weights");
    return true;
  }

  else if (responseName == "Solution Average") {
    responses[responseIndex] = Teuchos::rcp(new Albany::SolutionAverageResponseFunction());
    return true;
  }

  else if (responseName == "Solution Two Norm") {
    responses[responseIndex] = Teuchos::rcp(new Albany::SolutionTwoNormResponseFunction());
    return true;
  }

  else return false; // responseName not recognized
}

Teuchos::RCP<const Teuchos::ParameterList>
QCAD::SchrodingerProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidSchrodingerProblemParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Material", false, "");
  validPL->sublist("Potential", false, "");
  validPL->set<double>("EnergyUnitInElectronVolts",1e-3,"Energy unit in electron volts");
  validPL->set<double>("LengthUnitInMeters",1e-9,"Length unit in meters");

  validPL->sublist("Poisson Coupling", false, "");

  validPL->sublist("Poisson Coupling").set<bool>("Only solve in quantum blocks", false,"Only perform Schrodinger solve in element blocks marked as quatum regions.");
  validPL->sublist("Poisson Coupling").set<string>("Potential State Name", "","Name of State to use as potential");
  validPL->sublist("Poisson Coupling").set<int>("Save Eigenvectors as States", 0,"Number of eigenstates to save as states");
  return validPL;
}

