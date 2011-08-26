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


#include "QCAD_PoissonProblem.hpp"
#include "QCAD_MaterialDatabase.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"
#include "Albany_Utils.hpp"

QCAD::PoissonProblem::
PoissonProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
		const Teuchos::RCP<ParamLib>& paramLib_,
		const int numDim_,
		const Teuchos::RCP<const Epetra_Comm>& comm_) :
  Albany::AbstractProblem(params_, paramLib_, 1),
  comm(comm_),
  haveSource(false),
  numDim(numDim_)
{
  if (numDim==1) periodic = params->get("Periodic BC", false);
  else           periodic = false;
  if (periodic) *out <<" Periodic Boundary Conditions being used." <<std::endl;

  haveSource =  params->isSublist("Poisson Source");

  TEST_FOR_EXCEPTION(params->isSublist("Source Functions"), Teuchos::Exceptions::InvalidParameter,
		     "\nError! Poisson problem does not parse Source Functions sublist\n" 
                     << "\tjust Poisson Source sublist " << std::endl);

  //get length scale for problem (length unit for in/out mesh)
  length_unit_in_m = 1e-6; //default to um
  if(params->isType<double>("LengthUnitInMeters"))
    length_unit_in_m = params->get<double>("LengthUnitInMeters");

  temperature = 300; //default to 300K
  if(params->isType<double>("Temperature"))
    temperature = params->get<double>("Temperature");

  mtrlDbFilename = "materials.xml";
  if(params->isType<string>("MaterialDB Filename"))
    mtrlDbFilename = params->get<string>("MaterialDB Filename");

  //Schrodinger coupling
  nEigenvectors = 0;
  bUseSchrodingerSource = false;
  bUsePredictorCorrector = false;
  if(params->isSublist("Schrodinger Coupling")) {
    Teuchos::ParameterList& cList = params->sublist("Schrodinger Coupling");
    if(cList.isType<bool>("Schrodinger source in quantum blocks"))
      bUseSchrodingerSource = cList.get<bool>("Schrodinger source in quantum blocks");
    *out << "bSchod in quantum = " << bUseSchrodingerSource << std::endl;
    
    if(bUseSchrodingerSource && cList.isType<int>("Eigenvectors from States"))
      nEigenvectors = cList.get<int>("Eigenvectors from States");
    
    if(bUseSchrodingerSource && cList.isType<bool>("Use predictor-corrector method"))
      bUsePredictorCorrector = cList.get<bool>("Use predictor-corrector method");
  }

  *out << "Length unit = " << length_unit_in_m << " meters" << endl;
}

QCAD::PoissonProblem::
~PoissonProblem()
{
}

void
QCAD::PoissonProblem::
buildProblem(
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  constructEvaluators(meshSpecs, stateMgr, responses);
  constructDirichletEvaluators(meshSpecs);
}


void
QCAD::PoissonProblem::constructEvaluators(
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
   const int numVertices = cellType->getVertexCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << endl;

   RCP<DataLayout> shared_param = rcp(new MDALayout<Dim>(1));

   // Construct standard FEM evaluators with standard field names                              
   std::map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<Albany::Layouts> dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   Albany::ProblemUtils probUtils(dl);
   bool supportsTransient=false;

   // Define Field Names

   Teuchos::ArrayRCP<string> dof_names(neq);
     dof_names[0] = "Potential";

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

  { // Gather Eigenvectors
    RCP<ParameterList> p = rcp(new ParameterList);
    int type = FactoryTraits<AlbanyTraits>::id_gather_eigenvectors;
    p->set<int>("Type", type);
    p->set<string>("Eigenvector field name root", "Evec");
    p->set<int>("Number of eigenvectors", nEigenvectors);
    p->set< RCP<DataLayout> >("Data Layout", dl->node_scalar);

    evaluators_to_build["Gather Eigenvectors"] = p;
  }

  { // Permittivity
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_qcad_permittivity;
    p->set<int>("Type", type);

    p->set<string>("QP Variable Name", "Permittivity");
    p->set<string>("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set< RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Permittivity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    evaluators_to_build["Permittivity"] = p;
  }

  { // Temperature shared parameter (single scalar value, not spatially varying)
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_sharedparameter;
    p->set<int>("Type", type);

    p->set<string>("Parameter Name", "Temperature");
    p->set<double>("Parameter Value", temperature);
    p->set< RCP<DataLayout> >("Data Layout", shared_param);
    p->set< RCP<ParamLib> >("Parameter Library", paramLib);

    evaluators_to_build["Temperature"] = p;
  }

  if (haveSource) 
  { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    int type = FactoryTraits<AlbanyTraits>::id_qcad_poisson_source;
    p->set<int>("Type", type);

    //Input
    p->set< string >("Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Variable Name", "Potential");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::ParameterList& paramList = params->sublist("Poisson Source");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    //Output
    p->set<string>("Source Name", "Poisson Source");

    //Global Problem Parameters
    p->set<double>("Length unit in m", length_unit_in_m);
    p->set<string>("Temperature Name", "Temperature");
    p->set< RCP<DataLayout> >("Shared Param Data Layout", shared_param);
    p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

    // Schrodinger coupling
    p->set<bool>("Use Schrodinger source", bUseSchrodingerSource);
    p->set<int>("Schrodinger eigenvectors", nEigenvectors);
    p->set<string>("Eigenvector field name root", "Evec");
    p->set<bool>("Use predictor-corrector method", bUsePredictorCorrector);

    evaluators_to_build["Poisson Source"] = p;
  }

  // Interpolate Input Eigenvectors (if any) to quad points
  char buf[100];  
  for( int k = 0; k < nEigenvectors; k++)
  { 
    // DOF: Interpolate nodal Eigenvector values to quad points
    RCP<ParameterList> p;
    int type;

    //REAL PART
    sprintf(buf, "Poisson Eigenvector Re %d interpolate to qps", k);
    p = rcp(new ParameterList(buf));

    type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>   ("Type", type);

    // Input
    sprintf(buf, "Evec_Re%d", k);
    p->set<string>("Variable Name", buf);
    p->set< RCP<DataLayout> >("Node Data Layout",      dl->node_scalar);
    
    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    
    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    
    sprintf(buf, "Eigenvector Re %d interpolate to qps", k);
    evaluators_to_build[buf] = p;
    
    
    //IMAGINARY PART
    sprintf(buf, "Eigenvector Im %d interpolate to qps", k);
    p = rcp(new ParameterList(buf));
    
    type = FactoryTraits<AlbanyTraits>::id_dof_interpolation;
    p->set<int>   ("Type", type);
    
    // Input
    sprintf(buf, "Evec_Im%d", k);
    p->set<string>("Variable Name", buf);
    p->set< RCP<DataLayout> >("Node Data Layout",      dl->node_scalar);
    
    p->set<string>("BF Name", "BF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    
    // Output (assumes same Name as input)
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    
    sprintf(buf, "Eigenvector Im %d interpolate to qps", k);
    evaluators_to_build[buf] = p;
  }

  { // Potential Resid
    RCP<ParameterList> p = rcp(new ParameterList("Potential Resid"));

    int type = FactoryTraits<AlbanyTraits>::id_qcad_poisson_resid;
    p->set<int>("Type", type);

    //Input
    p->set<string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<string>("QP Variable Name", "Potential");

    p->set<string>("QP Time Derivative Variable Name", "Potential_dot");

    p->set<bool>("Have Source", haveSource);
    p->set<string>("Source Name", "Poisson Source");

    p->set<string>("Permittivity Name", "Permittivity");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Gradient QP Variable Name", "Potential Gradient");
    p->set<string>("Flux QP Variable Name", "Potential Flux");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<string>("Residual Name", "Potential Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    evaluators_to_build["Poisson Resid"] = p;
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

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::Residual>(res_tag);
   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::Jacobian>(jac_tag);
   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::Tangent>(tan_tag);
   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::SGResidual>(sgres_tag);
   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag);
   PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::SGTangent>(sgtan_tag);
   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::MPResidual>(mpres_tag);
   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag);
   PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag("Scatter", dl->dummy);
   fm->requireField<AlbanyTraits::MPTangent>(mptan_tag);

   //! Construct Responses
   Teuchos::ParameterList& responseList = params->sublist("Response Functions");
   Albany::ResponseUtils respUtils(dl);

   // Call local version of this function, instead of generic
   rfm = this->constructResponses(responses, responseList, evaluators_to_build, stateMgr, respUtils);
}

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
QCAD::PoissonProblem::constructResponses(
  std::vector< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses,
  Teuchos::ParameterList& responseList, 
  std::map<string, Teuchos::RCP<Teuchos::ParameterList> >& evaluators_to_build, 
  Albany::StateManager& stateMgr,
  Albany::ResponseUtils& respUtils)
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

void
QCAD::PoissonProblem::constructDirichletEvaluators(
     const Albany::MeshSpecsStruct& meshSpecs)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using std::map;
   using std::string;

   using PHAL::DirichletFactoryTraits;
   using PHAL::AlbanyTraits;

   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[0] = "Phi";   
   Albany::DirichletUtils dirUtils;

   const std::vector<std::string>& nodeSetIDs = meshSpecs.nsNames;

   Teuchos::ParameterList DBCparams = params->sublist("Dirichlet BCs");
   DBCparams.validateParameters(*(dirUtils.getValidDirichletBCParameters(nodeSetIDs,dirichletNames)),0); //TODO: Poisson version??

   // Create Material Database
   RCP<QCAD::MaterialDatabase> materialDB = rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));

   map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));
   vector<string> dbcs;

   // Check for all possible standard BCs (every dof on every nodeset) to see which is set
   for (std::size_t i=0; i<nodeSetIDs.size(); i++) {
     for (std::size_t j=0; j<dirichletNames.size(); j++) {

       std::stringstream sstrm; sstrm << "DBC on NS " << nodeSetIDs[i] << " for DOF " << dirichletNames[j];
       std::string ss = sstrm.str();

       if (DBCparams.isParameter(ss)) {
         RCP<ParameterList> p = rcp(new ParameterList);
         int type = DirichletFactoryTraits<AlbanyTraits>::id_qcad_poisson_dirichlet;
         p->set<int>("Type", type);

         p->set< RCP<DataLayout> >("Data Layout", dummy);
         p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", DBCparams.get<double>(ss));
         p->set< string >  ("Node Set ID", nodeSetIDs[i]);
         p->set< int >     ("Number of Equations", dirichletNames.size());
         p->set< int >     ("Equation Offset", j);

         p->set<RCP<ParamLib> >("Parameter Library", paramLib);

         //! Additional parameters needed for Poisson Dirichlet BCs
         Teuchos::ParameterList& paramList = params->sublist("Poisson Source");
         p->set<Teuchos::ParameterList*>("Poisson Source Parameter List", &paramList);
         //p->set<string>("Temperature Name", "Temperature");  //to add if use shared param for DBC
         p->set<double>("Temperature", temperature);
         p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);

         std::stringstream ess; ess << "Evaluator for " << ss;
         evaluators_to_build[ess.str()] = p;

         dbcs.push_back(ss);
       }
     }
   }

   //From here down, identical to Albany::AbstractProblem version of this function
   string allDBC="Evaluator for all Dirichlet BCs";
   {
      RCP<ParameterList> p = rcp(new ParameterList);
      int type = DirichletFactoryTraits<AlbanyTraits>::id_dirichlet_aggregator;
      p->set<int>("Type", type);

      p->set<vector<string>* >("DBC Names", &dbcs);
      p->set< RCP<DataLayout> >("Data Layout", dummy);
      p->set<string>("DBC Aggregator Name", allDBC);
      evaluators_to_build[allDBC] = p;
   }

   // Build Field Evaluators for each evaluation type
   PHX::EvaluatorFactory<AlbanyTraits,DirichletFactoryTraits<AlbanyTraits> > factory;
   RCP< vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > > evaluators;
   evaluators = factory.buildEvaluators(evaluators_to_build);

   // Create a DirichletFieldManager
   dfm = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

   // Register all Evaluators
   PHX::registerEvaluators(evaluators, *dfm);

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::Residual>(res_tag0);

   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::Jacobian>(jac_tag0);

   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::Tangent>(tan_tag0);

   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::SGResidual>(sgres_tag0);

   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag0);

   PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::SGTangent>(sgtan_tag0);

   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::MPResidual>(mpres_tag0);

   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag0);

   PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::MPTangent>(mptan_tag0);
}


Teuchos::RCP<const Teuchos::ParameterList>
QCAD::PoissonProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidPoissonProblemParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Permittivity", false, "");
  validPL->sublist("Poisson Source", false, "");
  validPL->set<double>("LengthUnitInMeters",1e-6,"Length unit in meters");
  validPL->set<double>("Temperature",300,"Temperature in Kelvin");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");

  validPL->sublist("Schrodinger Coupling", false, "");
  validPL->sublist("Schrodinger Coupling").set<bool>("Schrodinger source in quantum blocks",false,"Use eigenvector data to compute charge distribution within quantum blocks");
  validPL->sublist("Schrodinger Coupling").set<int>("Eigenvectors from States",0,"Number of eigenvectors to use for quantum region source");
  
  //For poisson schrodinger interations
  validPL->sublist("Dummy Dirichlet BCs", false, "");
  validPL->sublist("Dummy Parameters", false, "");

  return validPL;
}

