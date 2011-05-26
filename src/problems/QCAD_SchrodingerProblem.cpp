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
#include "Albany_BoundaryFlux1DResponseFunction.hpp"
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
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, 1),
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
  nEigenvectorsToOuputAsStates = 0;
  bOnlySolveInQuantumBlocks = false;

  //Poisson coupling
  if(params->isSublist("Poisson Coupling")) {
    Teuchos::ParameterList& cList = params->sublist("Poisson Coupling");
    if(cList.isType<bool>("Only solve in quantum blocks"))
      bOnlySolveInQuantumBlocks = cList.get<bool>("Only solve in quantum blocks");
    if(cList.isType<string>("Potential State Name"))
    potentialStateName = cList.get<string>("Potential State Name");

    if(cList.isType<int>("Save Eigenvectors as States"))
      nEigenvectorsToOuputAsStates = cList.get<int>("Save Eigenvectors as States");
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

  //STATES TO OUTPUT
  nstates = 1; // Potential energy
  nstates += nEigenvectorsToOuputAsStates*2; // Re and Im parts of each eigenvector
}

QCAD::SchrodingerProblem::
~SchrodingerProblem()
{
}

void
QCAD::SchrodingerProblem::
buildProblem(
    const int worksetSize,
    Albany::StateManager& stateMgr,
    const Albany::AbstractDiscretization& disc,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(worksetSize, disc.getCubatureDegree(), disc.getCellTopologyData(),stateMgr);
  constructDirichletEvaluators(disc.getNodeSetIDs());
 
  const Epetra_Map& dofMap = *(disc.getMap());
  int left_node = dofMap.MinAllGID();
  int right_node = dofMap.MaxAllGID();

  // Build response functions
  Teuchos::ParameterList& responseList = params->sublist("Response Functions");
  int num_responses = responseList.get("Number", 0);
  responses.resize(num_responses);
  for (int i=0; i<num_responses; i++) {
     std::string name = responseList.get(Albany::strint("Response",i), "??");

     if (name == "Boundary Flux 1D" && numDim==1) {
       // Need real size, not 1.0
       double h =  1.0 / (dofMap.NumGlobalElements() - 1);
       responses[i] =
         Teuchos::rcp(new Albany::BoundaryFlux1DResponseFunction(left_node,
                                                         right_node,
                                                         0, 1, h,
                                                         dofMap));
     }

     else if (name == "Solution Average")
       responses[i] = Teuchos::rcp(new Albany::SolutionAverageResponseFunction());

     else if (name == "Solution Two Norm")
       responses[i] = Teuchos::rcp(new Albany::SolutionTwoNormResponseFunction());

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
QCAD::SchrodingerProblem::constructEvaluators(
       const int worksetSize, const int cubDegree,
       const CellTopologyData& ctd,
       Albany::StateManager& stateMgr)
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

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = this->getIntrepidBasis(ctd);

   const int numNodes = intrepidBasis->getCardinality();

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, cubDegree);

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
   RCP<QCAD::MaterialDatabase> materialDB = rcp(new QCAD::MaterialDatabase(mtrlDbFilename));

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

    
    std::cout << "DEBUG: potential from state name " << potentialStateName << std::endl;

    // STATE INPUT / OUTPUT
    int issf = FactoryTraits<AlbanyTraits>::id_savestatefield;
    evaluators_to_build["Save Potential"] =
      stateMgr.registerStateVariable(potentialStateName, qp_scalar, dummy, issf);
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

   const Albany::StateManager::RegisteredStates& reg = stateMgr.getRegisteredStates();
   Albany::StateManager::RegisteredStates::const_iterator st = reg.begin();
   while (st != reg.end()) {
     PHX::Tag<AlbanyTraits::Residual::ScalarT> res_out_tag(st->first, dummy);
     fm->requireField<AlbanyTraits::Residual>(res_out_tag);
     st++;
   }

  // EIGENSTATE OUTPUT - to this registration after above loop because 
  //  we don't want evaluators to fill these states - they get filled 
  //  in SaveEigenData call after LOCA is done.
  if( nEigenvectorsToOuputAsStates > 0 ) {
    char evecStateName[100];
    for( int k = 0; k < nEigenvectorsToOuputAsStates; k++) {
      sprintf(evecStateName,"Eigenvector_Re%d",k);
      stateMgr.registerStateVariable(evecStateName, node_scalar);
      sprintf(evecStateName,"Eigenvector_Im%d",k);
      stateMgr.registerStateVariable(evecStateName, node_scalar);    
    }
  }

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

