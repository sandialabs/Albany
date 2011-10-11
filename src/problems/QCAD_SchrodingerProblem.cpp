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
  *out << "Energy unit = " << energy_unit_in_eV << " eV" << endl;

  length_unit_in_m = 1e-9; //default to nm
  if(params->isType<double>("LengthUnitInMeters"))
    length_unit_in_m = params->get<double>("LengthUnitInMeters");
  *out << "Length unit = " << length_unit_in_m << " meters" << endl;

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
}

QCAD::SchrodingerProblem::
~SchrodingerProblem()
{
}

void
QCAD::SchrodingerProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1); rfm.resize(1);
  constructEvaluators(*meshSpecs[0], stateMgr, responses);
  constructDirichletEvaluators(*meshSpecs[0]);
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
   using PHAL::FactoryTraits;
   using PHAL::AlbanyTraits;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

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

   // Construct standard FEM evaluators with standard field names                              

   std::map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   Albany::ProblemUtils probUtils(dl);
   bool supportsTransient=true;

   // Define Field Names

   Teuchos::ArrayRCP<string> dof_names(neq);
     dof_names[0] = "psi";

   Teuchos::ArrayRCP<string> dof_names_dot(neq);
   if (supportsTransient) {
     for (int i=0; i<neq; i++) dof_names_dot[i] = dof_names[i]+"_dot";
   }

   Teuchos::ArrayRCP<string> resid_names(neq);
     for (int i=0; i<neq; i++) resid_names[i] = dof_names[i]+" Residual";

   if (supportsTransient) evaluators_to_build["Gather Solution"] =
       probUtils.constructGatherSolutionEvaluator(false, dof_names, dof_names_dot);
   else  evaluators_to_build["Gather Solution"] =
       probUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names);

   evaluators_to_build["Scatter Residual"] =
     probUtils.constructScatterResidualEvaluator(false, resid_names);

   evaluators_to_build["Gather Coordinate Vector"] =
     probUtils.constructGatherCoordinateVectorEvaluator();

   evaluators_to_build["Map To Physical Frame"] =
     probUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature);

   evaluators_to_build["Compute Basis Functions"] =
     probUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);

   for (int i=0; i<neq; i++) {
     evaluators_to_build["DOF "+dof_names[i]] =
       probUtils.constructDOFInterpolationEvaluator(dof_names[i]);

     if (supportsTransient)
       evaluators_to_build["DOF "+dof_names_dot[i]] =
         probUtils.constructDOFInterpolationEvaluator(dof_names_dot[i]);

     evaluators_to_build["DOF Grad "+dof_names[i]] =
       probUtils.constructDOFGradInterpolationEvaluator(dof_names[i]);
  }

   // Create Material Database
   RCP<QCAD::MaterialDatabase> materialDB = rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));

  if (havePotential) { // Potential energy
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_schrodinger_potential;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "psi");
    p->set<string>("QP Potential Name", potentialStateName);
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Potential");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Global Problem Parameters
    p->set<double>("Energy unit in eV", energy_unit_in_eV);
    p->set<double>("Length unit in m", length_unit_in_m);

    evaluators_to_build["Potential Energy"] = p;

    //DEBUG: Output potential to check that it is imported properly; but 
    // this can be done via response evaluators now, so this code is no longer needed
    /*  *out << "DEBUG: potential from state name " << potentialStateName << std::endl->
    int issf = FactoryTraits<AlbanyTraits>::id_savestatefield;
    evaluators_to_build["Save Potential"] =
      stateMgr.registerStateVariable(potentialStateName, dl->qp_scalar, dl->dummy, issf);
    */
  }

  { // Wavefunction (psi) Resid
    RCP<ParameterList> p = rcp(new ParameterList("Wavefunction Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_schrodinger_resid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "psi");
    p->set<string>("QP Time Derivative Variable Name", "psi_dot");
    p->set<string>("QP Coordinate Vector Name", "Coord Vec");

    p->set<bool>("Have Potential", havePotential);
    p->set<bool>("Have Material", haveMaterial);
    p->set<string>("Potential Name", potentialStateName); // was "V"
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "psi Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
  
    //Output
    p->set<string>("Residual Name", "psi Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    if(haveMaterial) {
      Teuchos::ParameterList& paramList = params->sublist("Material");
      p->set<Teuchos::ParameterList*>("Material Parameter List", &paramList);
    }

    //Global Problem Parameters
    p->set<double>("Energy unit in eV", energy_unit_in_eV);
    p->set<double>("Length unit in m", length_unit_in_m);
    p->set<bool>("Only solve in quantum blocks", bOnlySolveInQuantumBlocks);
    p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    //Pass the Potential parameter list to test Finite Wall with different effective mass
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Potential");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    evaluators_to_build["psi Resid"] = p;
  }

   // Build Field Evaluators for each evaluation type
   PHX::EvaluatorFactory<AlbanyTraits,FactoryTraits<AlbanyTraits> > factory;
   RCP< vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > >
     evaluators;
   evaluators = factory.buildEvaluators(evaluators_to_build);

   // Create a FieldManager
   fm[0] = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

   // Register all Evaluators
   PHX::registerEvaluators(evaluators, *fm[0]);

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::Residual>(res_tag);
   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::Jacobian>(jac_tag);
   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::Tangent>(tan_tag);
   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::SGResidual>(sgres_tag);
   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
   PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::SGTangent>(sgtan_tag);
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::MPResidual>(mpres_tag);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
   PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag("Scatter", dl->dummy);
   fm[0]->requireField<AlbanyTraits::MPTangent>(mptan_tag);

   //! Construct Responses
   Teuchos::ParameterList& responseList = params->sublist("Response Functions");
   Albany::ResponseUtils respUtils(dl);

   // Call local version of this function, instead of generic
   rfm[0] = this->constructResponses(responses, responseList, evaluators_to_build, stateMgr, respUtils);
}

void
QCAD::SchrodingerProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[0] = "psi";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
QCAD::SchrodingerProblem::constructResponses(
  std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
  Teuchos::ParameterList& responseList, 
  std::map<string, Teuchos::RCP<Teuchos::ParameterList> >& evaluators_to_build, 
  Albany::StateManager& stateMgr, Albany::ResponseUtils& respUtils)
{
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using std::string;
  using PHAL::FactoryTraits;
  using PHAL::AlbanyTraits;
  Albany::Layouts& dl = *respUtils.get_dl();

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

     if( respUtils.getStdResponseFn(name, i, responseList, responses, stateMgr, p) ) {
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
       p->set< RCP<DataLayout> >("Dummy Data Layout", dl.dummy);
       
       p->set<int>("Type", type);
       p->set<string>("Coordinate Vector Name", "Coord Vec");
       p->set<string>("Weights Name",   "Weights");
       p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl.qp_scalar);
       p->set< RCP<DataLayout> >("QP Vector Data Layout", dl.qp_vector);

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
   return respUtils.createResponseFieldManager(response_evaluators_to_build, 
			      evaluators_to_build, responseIDs_to_require);
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

