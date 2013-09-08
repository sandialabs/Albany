//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "QCAD_Solver.hpp"
#include "QCAD_CoupledPoissonSchrodinger.hpp"
#include "Piro_Epetra_LOCASolver.hpp"
#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"
#include "Sacado_PCE_OrthogPoly.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

//needed?
//#include "Teuchos_RCP.hpp"
//#include "Teuchos_VerboseObject.hpp"
//#include "Teuchos_FancyOStream.hpp"

#include "Teuchos_ParameterList.hpp"

#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_EigendataInfoStruct.hpp"

#ifdef ALBANY_CI
#include "AnasaziConfigDefs.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziBlockDavidsonSolMgr.hpp"
#include "AnasaziBasicOutputManager.hpp"

#include "AlbanyCI_Types.hpp"
#include "AlbanyCI_Tensor.hpp"
#include "AlbanyCI_BlockTensor.hpp"
#include "AlbanyCI_SingleParticleBasis.hpp"
#include "AlbanyCI_BasisFactory.hpp"
#include "AlbanyCI_ManyParticleBasis.hpp"
#include "AlbanyCI_ManyParticleBasisBlock.hpp"
#include "AlbanyCI_MatrixFactory.hpp"
#include "AlbanyCI_ManyParticleMatrix.hpp"
#include "AlbanyCI_Solver.hpp"
#include "AlbanyCI_Solution.hpp"
#include "AlbanyCI_qnumbers.hpp"
#endif



namespace QCAD {
  
  void SolveModel(const SolverSubSolver& ss);
  void SolveModel(const SolverSubSolver& ss, 
		  Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates);
  void SolveModel(const QCAD::SolverSubSolver& ss, 
		  Teuchos::RCP<Albany::EigendataStruct>& pInitialEData, 
		  Teuchos::RCP<Albany::EigendataStruct>& pFinalEData);
  void SolveModel(const SolverSubSolver& ss, 
		  Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates,
		  Teuchos::RCP<Albany::EigendataStruct>& pInitialEData,
		  Teuchos::RCP<Albany::EigendataStruct>& pFinalEData);



  void CopyStateToContainer(Albany::StateArrays& src,
			    std::string stateNameToCopy,
			    std::vector<Intrepid::FieldContainer<RealType> >& dest);
  void CopyContainerToState(std::vector<Intrepid::FieldContainer<RealType> >& src,
			    Albany::StateArrays& dest,
			    std::string stateNameOfCopy);
  
  void CopyState(Albany::StateArrays& src, Albany::StateArrays& dest,  std::string stateNameToCopy);
  void AddStateToState(Albany::StateArrays& src, std::string srcStateNameToAdd, 
		       Albany::StateArrays& dest, std::string destStateNameToAddTo);
  void SubtractStateFromState(Albany::StateArrays& src, std::string srcStateNameToSubtract,
			      Albany::StateArrays& dest, std::string destStateNameToSubtractFrom);
  
  double getMaxDifference(Albany::StateArrays& states, 
			  std::vector<Intrepid::FieldContainer<RealType> >& prevState,
			  std::string stateName);
  
  void ResetEigensolverShift(const Teuchos::RCP<EpetraExt::ModelEvaluator>& Solver, double newShift,
			     Teuchos::RCP<Teuchos::ParameterList>& eigList);
  double GetEigensolverShift(const SolverSubSolver& ss, double pcBelowMinPotential);


  //String processing helper functions
  std::vector<std::string> string_split(const std::string& s, char delim, bool bProtect=false);
  std::string string_remove_whitespace(const std::string& s);
  std::vector<std::string> string_parse_function(const std::string& s);
  std::map<std::string,std::string> string_parse_arrayref(const std::string& s);
  std::vector<int> string_expand_compoundindex(const std::string& indexStr, int min_index, int max_index);

}



QCAD::Solver::
Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
       const Teuchos::RCP<const Epetra_Comm>& comm) :
  maxIter(0),CONVERGE_TOL(1e-6)
{
  using std::string;

  solverComm = comm;

  // Get sub-problem input xml files from problem parameters
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");

  // Validate Problem parameters against list for this specific problem
  problemParams.validateParameters(*getValidProblemParameters(),0);

  string problemName, problemDimStr;
  problemName = problemParams.get<string>("Name");
  problemNameBase = problemName.substr( 0, problemName.length()-3 ); //remove " xD" where x = 1, 2, or 3
  problemDimStr = problemName.substr( problemName.length()-2 ); // "xD" where x = 1, 2, or 3
  
  if(problemDimStr == "1D") numDims = 1;
  else if(problemDimStr == "2D") numDims = 2;
  else if(problemDimStr == "3D") numDims = 3;
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
				   << "Error!  Cannot extract dimension from problem name: "
				   << problemName << std::endl);

  if( !(problemNameBase == "Poisson" || problemNameBase == "Schrodinger" || problemNameBase == "Schrodinger CI" ||
	problemNameBase == "Poisson Schrodinger" || problemNameBase == "Poisson Schrodinger CI"))
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
				<< "Error!  Invalid problem base name: "
				<< problemNameBase << std::endl);
  

  // Check if "verbose" mode is enabled
  bVerbose = (comm->MyPID() == 0) && problemParams.get<bool>("Verbose Output", false);

  // Get problem parameters used for iterating Poisson-Schrodinger loop
  if(problemNameBase == "Poisson Schrodinger" || problemNameBase == "Poisson Schrodinger CI") {
    bUseIntegratedPS = problemParams.get<bool>("Use Integrated Poisson Schrodinger",true);
    maxIter = problemParams.get<int>("Maximum Iterations", 100);
    shiftPercentBelowMin = problemParams.get<double>("Eigensolver Percent Shift Below Potential Min", 1.0);
    CONVERGE_TOL = problemParams.get<double>("Iterative PS Convergence Tolerance", 1e-6);
  }

  // Get problem parameters used for Poisson-Schrodinger-CI mode
  if(problemNameBase == "Poisson Schrodinger CI") {
    maxCIParticles = problemParams.get<int>("Maximum CI Particles");
  }

  // Get problem parameters used for Schrodinger-CI mode
  if(problemNameBase == "Schrodinger CI") {
    nCIParticles = problemParams.get<int>("CI Particles");
    nCIExcitations = problemParams.get<int>("CI Excitations");
    assert(nCIParticles >= nCIExcitations);
  }

  // Get the number of eigenvectors - needed for all problems-modes except "Poisson"
  nEigenvectors = 0;
  if(problemNameBase != "Poisson") {
    nEigenvectors = problemParams.get<int>("Number of Eigenvalues");
  }

  // Get debug filenames -- empty string = don't output
  Teuchos::ParameterList& debugParams = appParams->sublist("Debug Output");
  std::string debug_initpoissonXML = debugParams.get<std::string>("Initial Poisson XML Input","");
  std::string debug_poissonXML     = debugParams.get<std::string>("Poisson XML Input","");
  std::string debug_schroXML       = debugParams.get<std::string>("Schrodinger XML Input","");
  std::string debug_psXML          = debugParams.get<std::string>("Poisson-Schrodinger XML Input","");
  std::string debug_initpoissonExo = debugParams.get<std::string>("Initial Poisson Exodus Output","");
  std::string debug_poissonExo     = debugParams.get<std::string>("Poisson Exodus Output","");
  std::string debug_schroExo       = debugParams.get<std::string>("Schrodinger Exodus Output","");

  // Get name of output exodus file specified in Discretization section
  std::string outputExo = appParams->sublist("Discretization").get<std::string>("Exodus Output File Name");

  /*OLD
  // Collect sub-problem input file names
  for(std::size_t i=0; i<nProblems; i++) {
    Teuchos::ParameterList& subProblemParams = problemParams.sublist(Albany::strint("Subproblem",i));
    string subName = subProblemParams.get<string>("Name");
    string inputFile = subProblemParams.get<string>("Input Filename");
    inputFilenames[subName] = inputFile;
    }*/

  std::string defaultSubSolver = "";

  // Create Solver parameter lists based on problem name
  if( problemNameBase == "Poisson" ) {
    subProblemAppParams["Poisson"] = createPoissonInputFile(appParams, numDims, nEigenvectors, "none",
							    debug_poissonXML, outputExo);
    defaultSubSolver = "Poisson";
  }

  else if( problemNameBase == "Schrodinger" ) {
    subProblemAppParams["Schrodinger"] = createSchrodingerInputFile(appParams, numDims, nEigenvectors, "none",
								    debug_schroXML, debug_schroExo);
    defaultSubSolver = "Schrodinger";
  }

  else if( problemNameBase == "Poisson Schrodinger" ) {
    subProblemAppParams["InitPoisson"] = createPoissonInputFile(appParams, numDims, nEigenvectors, "initial poisson",
								debug_initpoissonXML, debug_initpoissonExo);
    subProblemAppParams["Poisson"]     = createPoissonInputFile(appParams, numDims, nEigenvectors, "couple to schrodinger",
								debug_poissonXML, debug_poissonExo);
    subProblemAppParams["Schrodinger"] = createSchrodingerInputFile(appParams, numDims, nEigenvectors, "couple to poisson",
								    debug_schroXML, debug_schroExo);
    if(bUseIntegratedPS) {
      subProblemAppParams["PoissonSchrodinger"] = createPoissonSchrodingerInputFile(appParams, numDims, nEigenvectors,
										    debug_psXML, outputExo);
      defaultSubSolver = "PoissonSchrodinger";
    }
    else defaultSubSolver = "Poisson";    
  }

  else if( problemNameBase == "Schrodinger CI" ) {
    subProblemAppParams["CoulombPoisson"]   = createPoissonInputFile(appParams, numDims, nEigenvectors, "Coulomb",
								     debug_poissonXML, debug_poissonExo);
    subProblemAppParams["CoulombPoissonIm"] = createPoissonInputFile(appParams, numDims, nEigenvectors, "Coulomb imaginary",
								     "", ""); // no debug output
    subProblemAppParams["Schrodinger"] = createSchrodingerInputFile(appParams, numDims, nEigenvectors, "none",
								    debug_schroXML, debug_schroExo);
    defaultSubSolver = "Schrodinger";
  }

  else if( problemNameBase == "Poisson Schrodinger CI" ) {
    bUseIntegratedPS = false;  // TODO: add integrated end option to this -- (need to extract 1P eigenvectors from coupled PS solver below)

    subProblemAppParams["InitPoisson"] = createPoissonInputFile(appParams, numDims, nEigenvectors, "initial poisson",
								debug_initpoissonXML, debug_initpoissonExo);
    subProblemAppParams["Poisson"] = createPoissonInputFile(appParams, numDims, nEigenvectors, "couple to schrodinger",
							    debug_poissonXML, debug_poissonExo);
    subProblemAppParams["Schrodinger"] = createSchrodingerInputFile(appParams, numDims, nEigenvectors, "couple to poisson",
								    debug_schroXML, debug_schroExo);
    subProblemAppParams["CIPoisson"] = createPoissonInputFile(appParams, numDims, nEigenvectors, "CI", "", "");

    if(bUseIntegratedPS)
      subProblemAppParams["PoissonSchrodinger"] = createPoissonSchrodingerInputFile(appParams, numDims, nEigenvectors,
										    debug_psXML, outputExo);
    defaultSubSolver = "CIPoisson";

    // Note: no debug output for CI support poisson solvers in this mode
    subProblemAppParams["CoulombPoisson"]   = createPoissonInputFile(appParams, numDims, nEigenvectors, "Coulomb", "", "");
    subProblemAppParams["CoulombPoissonIm"] = createPoissonInputFile(appParams, numDims, nEigenvectors, "Coulomb imaginary", "", "");
    subProblemAppParams["NoChargePoisson"]  = createPoissonInputFile(appParams, numDims, nEigenvectors, "no charge", "", "");
    subProblemAppParams["DeltaPoisson"]     = createPoissonInputFile(appParams, numDims, nEigenvectors, "delta", "", "");
  }    

  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error in QCAD::Solver constructor:  " <<
				  "Invalid problem name base: " << problemNameBase << std::endl);

  //Create sub-solvers
  std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >::const_iterator itp;
  for(itp = subProblemAppParams.begin(); itp != subProblemAppParams.end(); ++itp) {
    const std::string& name = itp->first;
    const Teuchos::RCP<Teuchos::ParameterList>& param_list = itp->second;
    subSolvers[ name ] = CreateSubSolver( param_list , *comm);
  }

  //Create observer to create final exodus output file if necessary
  // (passing appParams to constructor will set output filename to that supplied by user)
  if(problemNameBase == "Schrodinger"   || 
     problemNameBase == "Schrodinger CI" ||
     (problemNameBase == "Poisson Schrodinger"    && !bUseIntegratedPS) ||
     (problemNameBase == "Poisson Schrodinger CI" && !bUseIntegratedPS  )) {
    final_obs = Teuchos::rcp(new QCAD::MultiSolution_Observer(getSubSolver(defaultSubSolver).app, appParams)); 
  }

  //Determine whether we should support DgDp (all sub-solvers must support DpDg for QCAD::Solver to)
  bSupportDpDg = true;  
  std::map<std::string, SolverSubSolver>::const_iterator it;
  for(it = subSolvers.begin(); it != subSolvers.end(); ++it) {
    EpetraExt::ModelEvaluator::OutArgs model_outargs = (it->second.model)->createOutArgs();
    if( model_outargs.supports(OUT_ARG_DgDp, 0, 0).none() ) { //just test if p=0, g=0 DgDp is supported
      bSupportDpDg = false;
      break;
    }
  }

  //Setup Parameter and responses maps
  
  // input file can have 
  //    <Parameter name="Parameter 0" type="string" value="Poisson[0]" />
  //    <Parameter name="Parameter 1" type="string" value="Poisson[1:3]" />
  //
  //    <Parameter name="Response 0" type="string" value="Poisson[0] # charge" />
  //    <Parameter name="Response 0" type="string" value="Schrodinger[1,3]" />
  //    <Parameter name="Response 0" type="string" value="=dist(Poisson[1:4],Poisson[4:7]) # distance example" />

  
  Teuchos::ParameterList& paramList = problemParams.sublist("Parameters");
  setupParameterMapping(paramList, defaultSubSolver);

  Teuchos::ParameterList& responseList = problemParams.sublist("Response Functions");
  setupResponseMapping(responseList, defaultSubSolver, nEigenvectors);

  num_p = (nParameters > 0) ? 1 : 0; // Only use first parameter (p) vector, if there are any parameters
  num_g = (responseFns.size() > 0) ? 1 : 0; // Only use first response vector (but really one more than num_g -- 2nd holds solution vector)


#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int GlobalIndex;
#else
  typedef long long GlobalIndex;
#endif

  // Create Epetra map for parameter vector (only one since num_p always == 1)
  epetra_param_map = Teuchos::rcp(new Epetra_LocalMap(static_cast<GlobalIndex>(nParameters), 0, *comm));

  // Create Epetra map for (first) response vector
  epetra_response_map = Teuchos::rcp(new Epetra_LocalMap(static_cast<GlobalIndex>(nResponseDoubles), 0, *comm));
     //ANDY: if (nResponseDoubles > 0) needed ??

  // Create Epetra map for solution vector (second response vector).  Assume 
  //  each subSolver has the same map, so just get the first one.
  Teuchos::RCP<const Epetra_Map> sub_x_map = (subSolvers.begin()->second).app->getMap();
  TEUCHOS_TEST_FOR_EXCEPT( sub_x_map == Teuchos::null );
  epetra_x_map = Teuchos::rcp(new Epetra_Map( *sub_x_map ));
}

QCAD::Solver::~Solver()
{
}


Teuchos::RCP<Teuchos::ParameterList> 
QCAD::Solver::createPoissonInputFile(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
				     int numDims, int nEigen, const std::string& specialProcessing,
				     const std::string& xmlOutputFile, const std::string& exoOutputFile) const
{
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  
  int vizDetail         = problemParams.get<int>("Phalanx Graph Visualization Detail",0);  
  double lenUnit        = problemParams.get<double>("LengthUnitInMeters", 1e-6);
  double energyUnit     = problemParams.get<double>("EnergyUnitInElectronVolts", 1.0);
  std::string matrlFile = problemParams.get<std::string>("MaterialDB Filename", "materials.xml");

  bool bXCPot  = problemParams.get<bool>("Include exchange-correlation potential",false);
  bool bQBOnly = problemParams.get<bool>("Only solve schrodinger in quantum blocks",true);
  bool bUsePCMethod = problemParams.get<bool>("Use predictor-corrector method",false);
  

  double Temp = -1;
  if(specialProcessing != "Coulomb" && specialProcessing != "Coulomb imaginary") { 
    Temp = problemParams.get<double>("Temperature"); //Temperature required for all poisson cases except "Coulomb" modes
  }

  // Get poisson & schrodinger problem sublists
  Teuchos::ParameterList& poisson_subList = problemParams.sublist("Poisson Problem", false);
  Teuchos::ParameterList& schro_subList = problemParams.sublist("Schrodinger Problem", false);
  
  // Create input parameter list for poission app which mimics a separate input file
  Teuchos::RCP<Teuchos::ParameterList> poisson_appParams = 
    Teuchos::createParameterList("Poisson Subapplication Parameters - " + specialProcessing);
  Teuchos::ParameterList& poisson_probParams = poisson_appParams->sublist("Problem",false);
  
  poisson_probParams.set("Name", QCAD::strdim("Poisson",numDims));
  poisson_probParams.set("Phalanx Graph Visualization Detail", vizDetail);
  poisson_probParams.set("LengthUnitInMeters",lenUnit);
  poisson_probParams.set("EnergyUnitInElectronVolts",energyUnit); 
  poisson_probParams.set("MaterialDB Filename", matrlFile);
  if(Temp >= 0) poisson_probParams.set("Temperature",Temp);

  // Poisson Source sublist processing
  {
    Teuchos::ParameterList auto_sourceList;
    auto_sourceList.set("Factor",1.0);
    auto_sourceList.set("Device","elementblocks");

    if(specialProcessing == "initial poisson") {
      auto_sourceList.set("Quantum Region Source", "semiclassical");
      auto_sourceList.set("Non Quantum Region Source", "semiclassical");

    } else if (specialProcessing == "couple to schrodinger") {
      auto_sourceList.set("Quantum Region Source", "schrodinger");
      auto_sourceList.set("Non Quantum Region Source", bQBOnly ? "semiclassical" : "schrodinger");
      auto_sourceList.set("Eigenvectors to Import", nEigen);
      auto_sourceList.set("Use predictor-corrector method", bUsePCMethod);
      auto_sourceList.set("Include exchange-correlation potential", bXCPot);

    } else if (specialProcessing == "Coulomb") {
      auto_sourceList.set("Quantum Region Source", "coulomb");
      auto_sourceList.set("Non Quantum Region Source", "none");
      auto_sourceList.set("Imaginary Part of Coulomb Source", false);
      auto_sourceList.set("Eigenvectors to Import", nEigen);

    } else if (specialProcessing == "Coulomb imaginary") {
      auto_sourceList.set("Quantum Region Source", "coulomb");
      auto_sourceList.set("Non Quantum Region Source", "none");
      auto_sourceList.set("Imaginary Part of Coulomb Source", true);
      auto_sourceList.set("Eigenvectors to Import", nEigen);
      auto_sourceList.set("Use predictor-corrector method", false);

    } else if (specialProcessing == "delta") {
      auto_sourceList.set("Quantum Region Source", "schrodinger");
      auto_sourceList.set("Non Quantum Region Source", "none");
      auto_sourceList.set("Eigenvectors to Import", nEigen);

    } else if (specialProcessing == "no charge") {
      auto_sourceList.set("Quantum Region Source", "none");
      auto_sourceList.set("Non Quantum Region Source", "none");
      auto_sourceList.set("Eigenvectors to Import", nEigen); //needed for responses

    } else if (specialProcessing == "CI") {
      auto_sourceList.set("Quantum Region Source", "ci");
      auto_sourceList.set("Non Quantum Region Source", "semiclassical");
      auto_sourceList.set("Eigenvectors to Import", nEigen);

    } else if (specialProcessing != "none") 
      TEUCHOS_TEST_FOR_EXCEPTION( true, Teuchos::Exceptions::InvalidParameter, 
				  "Invalid special processing for Poisson input: " << specialProcessing);
    
    Teuchos::ParameterList& sourceList = poisson_probParams.sublist("Poisson Source", false);
    if(poisson_subList.isSublist("Poisson Source"))
      sourceList.setParameters( poisson_subList.sublist("Poisson Source") );
    sourceList.setParametersNotAlreadySet( auto_sourceList );
  }

  // Permittivity sublist processing
  {
    Teuchos::ParameterList auto_permList;
    auto_permList.set("Permittivity Type","Block Dependent");

    Teuchos::ParameterList& permList = poisson_probParams.sublist("Permittivity", false);
    if(poisson_subList.isSublist("Permittivity"))
      permList.setParameters( poisson_subList.sublist("Permittivity") );
    permList.setParametersNotAlreadySet( auto_permList );
  }  


  // Dirichlet BC sublist processing
  if(poisson_subList.isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList& poisson_dbcList = poisson_probParams.sublist("Dirichlet BCs", false);
    poisson_dbcList.setParameters(poisson_subList.sublist("Dirichlet BCs"));
  }
  else if(schro_subList.isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList& poisson_dbcList = poisson_probParams.sublist("Dirichlet BCs", false);
    const Teuchos::ParameterList& schro_dbcList = schro_subList.sublist("Dirichlet BCs");
    Teuchos::ParameterList::ConstIterator it; double* dummy = NULL;
    for(it = schro_dbcList.begin(); it != schro_dbcList.end(); ++it) {
      std::string dbcName = schro_dbcList.name(it);
      std::size_t k = dbcName.find("psi");
      if( k != std::string::npos ) {
	dbcName.replace(k, 3 /* len("psi") */, "Phi");  // replace Phi -> psi
	poisson_dbcList.set( dbcName, schro_dbcList.entry(it).getValue(dummy) ); //copy all schrodinger DBCs
      }
    }
  }



  // Parameters sublist processing
  if( specialProcessing == "Coulomb" || specialProcessing == "Coulomb imaginary" ) {
    //! Add source eigenvector indices as parameters
    Teuchos::ParameterList& paramList = poisson_probParams.sublist("Parameters", false);
    if(poisson_subList.isSublist("Parameters"))
      paramList.setParameters(poisson_subList.sublist("Parameters"));
    
    int nParams = paramList.get<int>("Number", 0);
    paramList.set("Number", nParams + 2); //assumes Source Eigenvector X are not already params
    paramList.set(Albany::strint("Parameter",nParams), "Source Eigenvector 1");
    paramList.set(Albany::strint("Parameter",nParams+1), "Source Eigenvector 2");
  }
  else {
    Teuchos::ParameterList& poisson_paramsList = poisson_probParams.sublist("Parameters", false);
    if(poisson_subList.isSublist("Parameters"))
      poisson_paramsList.setParameters(poisson_subList.sublist("Parameters"));
    else poisson_paramsList.set("Number", 0);
  }


  // Reponse Functions sublist processing
  bool addCoulombIntegralResponses = ((specialProcessing == "delta") ||
				      (specialProcessing == "Coulomb") ||
				      (specialProcessing == "Coulomb imaginary") ||
				      (specialProcessing == "no charge") );
				 
  if(addCoulombIntegralResponses) 
  {
    Teuchos::ParameterList& responseList = poisson_probParams.sublist("Response Functions", false);
    if(poisson_subList.isSublist("Response Functions"))
      responseList.setParameters(poisson_subList.sublist("Response Functions"));

    //! Add responses for each pair ( evec_i, evec_j )
    int initial_nResponses = responseList.get<int>("Number",0); //Shift existing responses
    int added_nResponses = 2 * nEigen * (nEigen + 1) / 2;  //mult by 2 for real & imag parts
    char buf1[200], buf2[200], buf1i[200], buf2i[200];
    responseList.set("Number", initial_nResponses + added_nResponses);

    //shift response indices of existing responses by added_responses so added responses index from zero
    for(int i=initial_nResponses-1; i >= 0; i--) {       
      std::string respType = responseList.get<std::string>(Albany::strint("Response",i));
      responseList.set(Albany::strint("Response",i + added_nResponses), respType); //shift response index
      if(responseList.isSublist( Albany::strint("ResponseParams",i) )) {            //shift response params index (if applicable)
	responseList.sublist( Albany::strint("ResponseParams",i + added_nResponses) ) = 
	  Teuchos::ParameterList(responseList.sublist( Albany::strint("ResponseParams",i) ) ); //create new copy of list
	responseList.sublist( Albany::strint("ResponseParams",i) ) = Teuchos::ParameterList(Albany::strint("ResponseParams",i)); //clear sublist i
      }
    }

    int iResponse = 0;
    for(int i=0; i<nEigenvectors; i++) {
      sprintf(buf1, "%s_Re%d", "Evec", i);
      sprintf(buf1i, "%s_Im%d", "Evec", i);
      for(int j=i; j<nEigenvectors; j++) {
	sprintf(buf2, "%s_Re%d", "Evec", j);
	sprintf(buf2i, "%s_Im%d", "Evec", j);

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams.set("Field Name", "Electric Potential");  // same as solution, but must be at quad points
	responseParams.set("Field Name 1", buf1);  responseParams.set("Field Name Im 1", buf1i);
	responseParams.set("Field Name 2", buf2);  responseParams.set("Field Name Im 2", buf2i);
	responseParams.set("Conjugate Field 1", true);
	responseParams.set("Conjugate Field 2", false);
	responseParams.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams.set("Return Imaginary Part", false);

	iResponse++;

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams2 = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams2.set("Field Name", "Electric Potential");  // same as solution, but must be at quad points
	responseParams2.set("Field Name 1", buf1);  responseParams2.set("Field Name Im 1", buf1i);
	responseParams2.set("Field Name 2", buf2);  responseParams2.set("Field Name Im 2", buf2i);
	responseParams2.set("Conjugate Field 1", true);
	responseParams2.set("Conjugate Field 2", false);
	responseParams2.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams2.set("Return Imaginary Part", true);

	iResponse++;
      }
    }
  }
  else if(specialProcessing == "couple to schrodinger" || specialProcessing == "initial poisson" || specialProcessing == "CI")
  {
    // Assume user has not already added the responses needed to couple with a schrodinger solver, so add them here
    Teuchos::ParameterList& responseList = poisson_probParams.sublist("Response Functions", false);
    if(poisson_subList.isSublist("Response Functions"))
      responseList.setParameters(poisson_subList.sublist("Response Functions"));

    int nResponses = responseList.get<int>("Number",0);
    int nAddedResponses = 6;
    responseList.set("Number", nResponses+nAddedResponses);

    int iResponse = nResponses;
    Teuchos::ParameterList* pResponseParams;

    // Add responses to save the Electric Potential, Conduction Band, and Potential for use in the P-S iterations
    responseList.set(Albany::strint("Response",iResponse), "Save Field");
    pResponseParams = &responseList.sublist(Albany::strint("ResponseParams",iResponse));
    pResponseParams->set("Field Name", "Electric Potential");
    pResponseParams->set("State Name", "PS Saved Electric Potential");
    pResponseParams->set("Output Cell Average", false);
    pResponseParams->set("Output to Exodus", false);
    iResponse++;

    responseList.set(Albany::strint("Response",iResponse), "Save Field");
    pResponseParams = &responseList.sublist(Albany::strint("ResponseParams",iResponse));
    pResponseParams->set("Field Name", "Conduction Band");
    pResponseParams->set("State Name", "PS Conduction Band");
    pResponseParams->set("Output Cell Average", false);
    pResponseParams->set("Output to Exodus", false);
    iResponse++;

    responseList.set(Albany::strint("Response",iResponse), "Save Field");
    pResponseParams = &responseList.sublist(Albany::strint("ResponseParams",iResponse));
    pResponseParams->set("Field Name", "Potential");
    pResponseParams->set("State Name", "PS Saved Solution");
    pResponseParams->set("Output Cell Average", false);
    pResponseParams->set("Output to Exodus", false);
    iResponse++;

    // Add dummy response to "save" into "PS Previous Poisson Potential" state so that memory is allocated 
    //  within the state manager for this state.
    responseList.set(Albany::strint("Response",iResponse), "Save Field");
    pResponseParams = &responseList.sublist(Albany::strint("ResponseParams",iResponse));
    pResponseParams->set("Field Name", "Potential");
    pResponseParams->set("State Name", "PS Previous Poisson Potential");
    pResponseParams->set("Output Cell Average", false);
    pResponseParams->set("Output to Exodus", false);
    pResponseParams->set("Memory Placeholder Only", true);
    iResponse++;

    // SECOND TO LAST RESPONSE: compute the total number of electrons in quantum regions (used for CI runs) 
    responseList.set(Albany::strint("Response",iResponse), "Field Integral");
    pResponseParams = &responseList.sublist(Albany::strint("ResponseParams",iResponse));
    pResponseParams->set("Type", "Field Integral"); // TODO: is this needed?
    pResponseParams->set("Field Name", "Electron Density");
    pResponseParams->set("Quantum Element Blocks Only", true);
    iResponse++;

    // LAST added response: compute the minimum of conduction band in the quantum regions.  This MUST be
    //  the last response, since the P-S loop expects it there (and uses it to compute the shift for the schrodinger eigensolver)
    // Note: it used to be the first response, but this would require shifting user responses, and would mess with indices
    responseList.set(Albany::strint("Response",iResponse), "Field Value");
    pResponseParams = &responseList.sublist(Albany::strint("ResponseParams",iResponse));
    pResponseParams->set("Operation", "Minimize");
    pResponseParams->set("Operation Field Name", "Conduction Band");
    pResponseParams->set("Quantum Element Blocks Only", true);

  }
  else {
    Teuchos::ParameterList& poisson_respList = poisson_probParams.sublist("Response Functions", false);
    if(poisson_subList.isSublist("Response Functions"))
      poisson_respList.setParameters(poisson_subList.sublist("Response Functions"));
    else poisson_respList.set("Number", 0);
  }  

  // Discretization sublist processing
  Teuchos::ParameterList& discList = appParams->sublist("Discretization");
  Teuchos::ParameterList& poisson_discList = poisson_appParams->sublist("Discretization", false);
  poisson_discList.setParameters(discList);
  if(exoOutputFile.length() > 0) 
    poisson_discList.set("Exodus Output File Name",exoOutputFile);
  else poisson_discList.remove("Exodus Output File Name",false); 

  // Piro sublist processing
  Teuchos::ParameterList& poisson_piroList = poisson_appParams->sublist("Piro", false);
  poisson_piroList.setParameters( appParams->sublist("Piro") ); // copy Piro list from app
  poisson_piroList.sublist("Analysis").sublist("Solve").set("Compute Sensitivities", false); // don't compute sensitivities

  if(xmlOutputFile.length() > 0 && solverComm->MyPID() == 0)
    Teuchos::writeParameterListToXmlFile(*poisson_appParams, xmlOutputFile);

  return poisson_appParams;
}


Teuchos::RCP<Teuchos::ParameterList> 
QCAD::Solver::createSchrodingerInputFile(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
					 int numDims, int nEigen, const std::string& specialProcessing,
					 const std::string& xmlOutputFile, const std::string& exoOutputFile) const
{
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  
  int vizDetail         = problemParams.get<int>("Phalanx Graph Visualization Detail",0);  
  double lenUnit        = problemParams.get<double>("LengthUnitInMeters", 1e-6);
  double energyUnit     = problemParams.get<double>("EnergyUnitInElectronVolts", 1.0);
  std::string matrlFile = problemParams.get<std::string>("MaterialDB Filename", "materials.xml");

  // Only used by schrodinger, poisson-schrodinger, and poisson-schroinger-ci solvers, but has default
  bool bQBOnly = problemParams.get<bool>("Only solve schrodinger in quantum blocks",true);

  // Get poisson and schrodinger problem sublists
  Teuchos::ParameterList& schro_subList = problemParams.sublist("Schrodinger Problem", false);
  Teuchos::ParameterList& poisson_subList = problemParams.sublist("Poisson Problem", false);


  Teuchos::RCP<Teuchos::ParameterList> schro_appParams = 
    Teuchos::createParameterList("Schrodinger Subapplication Parameters");
  Teuchos::ParameterList& schro_probParams = schro_appParams->sublist("Problem",false);

  schro_probParams.set("Name", QCAD::strdim("Schrodinger",numDims));
  schro_probParams.set("Solution Method", "Continuation");
  schro_probParams.set("Phalanx Graph Visualization Detail", vizDetail);
  schro_probParams.set("EnergyUnitInElectronVolts",energyUnit);
  schro_probParams.set("LengthUnitInMeters",lenUnit);
  schro_probParams.set("MaterialDB Filename", matrlFile);

  // Poisson Coupling sublist processing
  if(specialProcessing == "couple to poisson")
  {    
    Teuchos::ParameterList auto_couplingList;
    auto_couplingList.set("Only solve in quantum blocks", bQBOnly);
    auto_couplingList.set("Potential State Name", "PS Conduction Band"); 
       //import potential from "PS Conduction Band" state to a *field* with the same name.
    //auto_couplingList.set("Save Eigenvectors as States", nEigenvals); //DEPRECATED - remove this

    Teuchos::ParameterList& couplingList = schro_probParams.sublist("Poisson Coupling", false); //TODO -remove this sublist all together --> Potential
    if(schro_subList.isSublist("Poisson Coupling"))
      couplingList.setParameters(schro_subList.sublist("Poisson Coupling"));
    couplingList.setParametersNotAlreadySet(auto_couplingList);
  }
  else if(schro_subList.isSublist("Poisson Coupling")) {
    schro_probParams.sublist("Poisson Coupling", false).setParameters(schro_subList.sublist("Poisson Coupling"));
  }

  // Potential sublist processing
  if(specialProcessing == "couple to poisson")
  {
    Teuchos::ParameterList auto_potList;
    auto_potList.set("Type","From State");
    auto_potList.set("Scaling Factor",1.0);
    
    Teuchos::ParameterList& potList = schro_probParams.sublist("Potential", false);
    if(schro_subList.isSublist("Potential"))
      potList.setParameters(schro_subList.sublist("Potential"));
    potList.setParametersNotAlreadySet(auto_potList);
  }
  else if(schro_subList.isSublist("Potential")) {
    schro_probParams.sublist("Potential", false).setParameters(schro_subList.sublist("Potential"));
  }


  // Dirichlet BC sublist processing
  if(schro_subList.isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList& schro_dbcList = schro_probParams.sublist("Dirichlet BCs", false);
    schro_dbcList.setParameters(schro_subList.sublist("Dirichlet BCs"));
  }
  else if(poisson_subList.isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList& schro_dbcList = schro_probParams.sublist("Dirichlet BCs", false);
    const Teuchos::ParameterList& poisson_dbcList = poisson_subList.sublist("Dirichlet BCs");
    Teuchos::ParameterList::ConstIterator it;
    for(it = poisson_dbcList.begin(); it != poisson_dbcList.end(); ++it) {
      std::string dbcName = poisson_dbcList.name(it);
      std::size_t k = dbcName.find("Phi");
      if( k != std::string::npos ) {
	dbcName.replace(k, 3 /* len("Phi") */, "psi");  // replace Phi -> psi
	schro_dbcList.set( dbcName, 0.0 ); //copy all poisson DBCs but set to zero
      }
    }
  }

  // Parameters sublist processing -- ensure "Schrodinger Potential Scaling Factor" 
  //   appears in list, since this is needed by LOCA continuation analysis
  {
    Teuchos::ParameterList& paramsList = schro_probParams.sublist("Parameters", false);    
    if(schro_subList.isSublist("Parameters"))
      paramsList.setParameters(schro_subList.sublist("Parameters"));

    bool bAddScalingFactor = true;
    Teuchos::ParameterList::ConstIterator it; std::string* dummy = NULL;
    for(it = paramsList.begin(); it != paramsList.end(); ++it) {
      if(paramsList.entry(it).isType<std::string>() &&
	 paramsList.entry(it).getValue(dummy) == "Schrodinger Potential Scaling Factor") { 
	bAddScalingFactor = false; break; 
      }
    }

    if(bAddScalingFactor) {
      int nParams = paramsList.get<int>("Number",0);
      paramsList.set("Number", nParams+1);
      paramsList.set( Albany::strint("Parameter",nParams), "Schrodinger Potential Scaling Factor" );
    }
  }

  // Response Functions sublist processing
  if(specialProcessing == "couple to poisson")
  {
    // Assume user has not already added the responses needed to couple with a poisson solver, so add them here
    Teuchos::ParameterList& responseList = schro_probParams.sublist("Response Functions", false);
    if(schro_subList.isSublist("Response Functions"))
      responseList.setParameters(schro_subList.sublist("Response Functions"));

    int nResponses = responseList.get<int>("Number",0);
    responseList.set("Number", nResponses+2);

      // First added response: dummy "save" into "PS Conduction Band" state so memory is allocated
      //  within the state manager for this state.  The state gets filled prior to solving the schrodinger problem
      //  via the state manager's importing it, and then the state is used as the potential energy in the schro. eqn.
    responseList.set(Albany::strint("Response",nResponses), "Save Field");
    Teuchos::ParameterList& responseParams1 = responseList.sublist(Albany::strint("ResponseParams",nResponses));
    responseParams1.set("Field Name", "PS Conduction Band"); //Field name given by "Potential State Name" above TODO: change this to always be "Potential" or "V"??
    responseParams1.set("State Name", "PS Conduction Band");
    responseParams1.set("Output Cell Average", false);
    responseParams1.set("Output to Exodus", false);
    responseParams1.set("Memory Placeholder Only", true);

      // Second added response: dummy "save" into "PS Previous Poisson Potential" state so that memory is allocated 
      //  within the state manager for this state. (see Poisson-Schrodigner iteration code)
    responseList.set(Albany::strint("Response",nResponses+1), "Save Field");
    Teuchos::ParameterList& responseParams2 = responseList.sublist(Albany::strint("ResponseParams",nResponses+1));
    responseParams2.set("Field Name", "PS Conduction Band");
    responseParams2.set("State Name", "PS Previous Poisson Potential");
    responseParams2.set("Output Cell Average", false);
    responseParams2.set("Output to Exodus", false);
    responseParams2.set("Memory Placeholder Only", true);
  }
  else {
    Teuchos::ParameterList& schro_respList = schro_probParams.sublist("Response Functions", false);
    if(schro_subList.isSublist("Response Functions"))
      schro_respList.setParameters(schro_subList.sublist("Response Functions"));
    else schro_respList.set("Number", 0);
  }

    // Material sublist processing
  if(schro_subList.isSublist("Material")) {
    Teuchos::ParameterList& schro_mtrlList = schro_probParams.sublist("Material", false);
    schro_mtrlList.setParameters(schro_subList.sublist("Material"));
  }


    // Discretization sublist processing
  Teuchos::ParameterList& discList = appParams->sublist("Discretization");
  Teuchos::ParameterList& schro_discList = schro_appParams->sublist("Discretization", false);
  schro_discList.setParameters(discList);
  if(exoOutputFile.length() > 0) 
    schro_discList.set("Exodus Output File Name",exoOutputFile);
  else schro_discList.remove("Exodus Output File Name",false); 

    // Piro sublist processing
  Teuchos::ParameterList& schro_piroList = schro_appParams->sublist("Piro", false);
  schro_piroList.setParameters( appParams->sublist("Piro") ); // copy Piro list from app
  //schro_piroList.sublist("Analysis").sublist("Solve").set("Compute Sensitivities", false); // don't compute sensitivities
    // NOTE: not computing sensitivities messes with LOCA eigensolver somehow and causes different (I think correct) results...
    //  maybe just no mass matrix renorm - no, not that... ANDY - why are eigenvectors different (evals are the same) when 
    //  Compute Sensitivities == false vs. true (seems like 'true' gives correct evecs...)

  //if(schro_piroList.sublist("Piro").sublist("NOX").sublist("Direction").sublist("Newton").isSublist("Linear Solver"))
  //{
  //  schro_piroList.sublist("Piro").sublist("NOX").sublist("Direction").sublist("Newton").remove("Linear Solver");
  //}


  if(xmlOutputFile.length() > 0 && solverComm->MyPID() == 0)
    Teuchos::writeParameterListToXmlFile(*schro_appParams, xmlOutputFile);

  return schro_appParams;
}


Teuchos::RCP<Teuchos::ParameterList> 
QCAD::Solver::createPoissonSchrodingerInputFile(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
						int numDims, int nEigen, const std::string& xmlOutputFile,
						const std::string& exoOutputFile) const
{
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  
  int vizDetail         = problemParams.get<int>("Phalanx Graph Visualization Detail");  
  double lenUnit        = problemParams.get<double>("LengthUnitInMeters", 1e-6);
  std::string matrlFile = problemParams.get<std::string>("MaterialDB Filename", "materials.xml");

  bool bXCPot  = problemParams.get<bool>("Include exchange-correlation potential",false);
  bool bQBOnly = problemParams.get<bool>("Only solve schrodinger in quantum blocks",true);

  double Temp = problemParams.get<double>("Temperature");

  // Get poisson & schodinger problem sublists
  Teuchos::ParameterList& poisson_subList = problemParams.sublist("Poisson Problem", false);
  Teuchos::ParameterList& schro_subList   = problemParams.sublist("Schrodigner Problem", false);
  
  // Create input parameter list for poission app which mimics a separate input file
  Teuchos::RCP<Teuchos::ParameterList> ps_appParams = 
    Teuchos::createParameterList("Poisson-Schrodinger Subapplication Parameters");
  Teuchos::ParameterList& ps_probParams = ps_appParams->sublist("Problem",false);

  ps_probParams.set("Solution Method", "QCAD Poisson-Schrodinger");  
  ps_probParams.set("Name", QCAD::strdim("Poisson Schrodinger",numDims));
  ps_probParams.set("Phalanx Graph Visualization Detail", vizDetail);
  ps_probParams.set("LengthUnitInMeters",lenUnit);
  ps_probParams.set("MaterialDB Filename", matrlFile);
  ps_probParams.set("Temperature",Temp);
  ps_probParams.set("Number of Eigenvalues",nEigen);
  ps_probParams.set("Verbose Output", true);

  ps_probParams.set("Include exchange-correlation potential", bXCPot);
  ps_probParams.set("Only solve schrodinger in quantum blocks", bQBOnly);

  // Poisson Problem sublist processing
  Teuchos::ParameterList& ps_poissonParams = ps_probParams.sublist("Poisson Problem",false);

  if(poisson_subList.isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList& tmp = ps_poissonParams.sublist("Dirichlet BCs", false);
    tmp.setParameters(poisson_subList.sublist("Dirichlet BCs"));
  }

  if(poisson_subList.isSublist("Parameters")) {
    Teuchos::ParameterList& tmp = ps_poissonParams.sublist("Parameters", false);
    tmp.setParameters(poisson_subList.sublist("Parameters"));
  }

  if(poisson_subList.isSublist("Response Functions")) {
    Teuchos::ParameterList& tmp = ps_poissonParams.sublist("Response Functions", false);
    tmp.setParameters(poisson_subList.sublist("Response Functions"));
  }


  // Schrodinger Problem sublist processing
  Teuchos::ParameterList& ps_schroParams = ps_probParams.sublist("Schrodinger Problem",false);

  // copy Parameters, Dirichlet BCs, and Responses sublists from schro_subList if they're present
  if(schro_subList.isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList& tmp = ps_schroParams.sublist("Dirichlet BCs", false);
    tmp.setParameters(schro_subList.sublist("Dirichlet BCs"));
  }

  if(schro_subList.isSublist("Parameters")) {
    Teuchos::ParameterList& tmp = ps_schroParams.sublist("Parameters", false);
    tmp.setParameters(schro_subList.sublist("Parameters"));
  }

  if(schro_subList.isSublist("Response Functions")) {
    Teuchos::ParameterList& tmp = ps_schroParams.sublist("Response Functions", false);
    tmp.setParameters(schro_subList.sublist("Response Functions"));
  }


  // Debug Output sublist processing
  if(appParams->isSublist("Debug Output")) {
    Teuchos::ParameterList& debugParams = appParams->sublist("Debug Output");
    std::string poissonXML = debugParams.get<std::string>("PS Poisson XML Input","");
    std::string schroXML   = debugParams.get<std::string>("PS Schrodinger XML Input","");
    std::string poissonExo = debugParams.get<std::string>("PS Poisson Exodus Output","");
    std::string schroExo   = debugParams.get<std::string>("PS Schrodinger Exodus Output","");

    Teuchos::ParameterList& ps_debugParams = ps_appParams->sublist("Debug Output");
    if(poissonXML.length() > 0) ps_debugParams.set("Poisson XML Input", poissonXML);
    if(schroXML.length() > 0)   ps_debugParams.set("Schrodinger XML Input", schroXML);
    if(poissonExo.length() > 0) ps_debugParams.set("Poisson Exodus Output", poissonExo);
    if(schroExo.length() > 0)   ps_debugParams.set("Schrodinger Exodus Output", schroExo);
  }
  
  // Discretization sublist processing
  Teuchos::ParameterList& discList = appParams->sublist("Discretization");
  Teuchos::ParameterList& ps_discList = ps_appParams->sublist("Discretization", false);
  ps_discList.setParameters(discList);
  if(exoOutputFile.length() > 0) 
    ps_discList.set("Exodus Output File Name",exoOutputFile);
  else ps_discList.remove("Exodus Output File Name",false); 

  
  // Piro sublist processing
  Teuchos::ParameterList& ps_piroList = ps_appParams->sublist("Piro", false);
  ps_piroList.setParameters( appParams->sublist("Piro") ); // copy Piro list from app
  ps_piroList.set("Solver Type", "NOX");  //note: not automatically filled in by SolverFactory

  if(xmlOutputFile.length() > 0 && solverComm->MyPID() == 0)
    Teuchos::writeParameterListToXmlFile(*ps_appParams, xmlOutputFile);

  return ps_appParams;
}

Teuchos::RCP<const Epetra_Map> QCAD::Solver::get_x_map() const
{
  Teuchos::RCP<const Epetra_Map> neverused;
  return neverused;
}

Teuchos::RCP<const Epetra_Map> QCAD::Solver::get_f_map() const
{
  Teuchos::RCP<const Epetra_Map> neverused;
  return neverused;
}

Teuchos::RCP<const Epetra_Map> QCAD::Solver::get_p_map(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l >= num_p || l < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in QCAD::Solver::get_p_map():  " <<
                     "Invalid parameter index l = " <<
                     l << std::endl);

  return epetra_param_map;  //no index because num_p == 1 so l must be zero
}

Teuchos::RCP<const Epetra_Map> QCAD::Solver::get_g_map(int j) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(j > num_g || j < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in QCAD::Solver::get_g_map():  " <<
                     "Invalid response index j = " <<
                     j << std::endl);

  if      (j < num_g) return epetra_response_map;  //no index because num_g == 1 so j must be zero
  else if (j == num_g) return epetra_x_map;
  return Teuchos::null;
}

Teuchos::RCP<const Epetra_Vector> QCAD::Solver::get_x_init() const
{
  Teuchos::RCP<const Epetra_Vector> neverused;
  return neverused;
}

Teuchos::RCP<const Epetra_Vector> QCAD::Solver::get_p_init(int l) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(l >= num_p || l < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in QCAD::Solver::get_p_init():  " <<
                     "Invalid parameter index l = " <<
                     l << std::endl);

  Teuchos::RCP<Epetra_Vector> p_init = 
    Teuchos::rcp(new Epetra_Vector(*(epetra_param_map)));

  // Take initial value from the first (if multiple) parameter 
  //    fns for each given parameter
  for(std::size_t i=0; i<nParameters; i++) {
    (*p_init)[i] = paramFnVecs[i][0]->getInitialParam(subSolvers);
  }

  return p_init;
}

EpetraExt::ModelEvaluator::InArgs QCAD::Solver::createInArgs() const
{
  EpetraExt::ModelEvaluator::InArgsSetup inArgs;
  inArgs.setModelEvalDescription("QCAD Solver Model Evaluator Description");
  inArgs.set_Np(num_p);
  return inArgs;
}

EpetraExt::ModelEvaluator::OutArgs QCAD::Solver::createOutArgs() const
{
  //Based on Piro_Epetra_NOXSolver.cpp implementation
  EpetraExt::ModelEvaluator::OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("QCAD Solver Multipurpose Model Evaluator");
  // Ng is 1 bigger then model-Ng so that the solution vector can be an outarg
  outArgs.set_Np_Ng(num_p, num_g+1);  //TODO: is the +1 necessary still??

  // TODO -- make this more robust... no hardcoded names...
  std::string refSolverName;
  if(subSolvers.count("Poisson") > 0) refSolverName = "Poisson";
  else if(subSolvers.count("Schrodinger") > 0) refSolverName = "Schrodinger";
  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       "QCAD::Solver could not find a reference solver for this problem." << std::endl);
  
  const SolverSubSolver& referenceSolver = getSubSolver(refSolverName);

  // We support all dg/dp layouts model supports, plus the linear op layout
  EpetraExt::ModelEvaluator::OutArgs model_outargs = referenceSolver.model->createOutArgs();
  for (int i=0; i<num_g; i++) {
    for (int j=0; j<num_p; j++) {
      DerivativeSupport ds = model_outargs.supports(OUT_ARG_DgDp, i, j);
      if (!ds.none()) {
	ds.plus(DERIV_LINEAR_OP);
	outArgs.setSupports(OUT_ARG_DgDp, i, j, ds);
      }
    }
  }

  /*OLD
  //Derivative info 
  if(bSupportDpDg) {
    for (int i=0; i<num_g; i++) {
      for (int j=0; j<num_p; j++)
	outArgs.setSupports(OUT_ARG_DgDp, i, j, DerivativeSupport(DERIV_MV_BY_COL));
    }
  }*/

  return outArgs;
}

void 
QCAD::Solver::evalModel(const InArgs& inArgs,
			const OutArgs& outArgs ) const
{
  std::vector<double> eigenvalueResponses;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  // update sub-solver parameters using the main solver's parameter values
  if(num_p > 0) {   // or could use: (inArgs.Np() > 0)
    Teuchos::RCP<const Epetra_Vector> p = inArgs.get_p(0); //only use *first* param vector
    std::vector<Teuchos::RCP<QCAD::SolverParamFn> >::const_iterator pit;
    for(std::size_t i=0; i<nParameters; i++) {
      for(pit = paramFnVecs[i].begin(); pit != paramFnVecs[i].end(); pit++) {
	(*pit)->fillSubSolverParams((*p)[i], subSolvers);
      }
    }

    if(bVerbose) {
      *out << "BEGIN QCAD Solver Parameters:" << std::endl;
      for(std::size_t i=0; i<nParameters; i++)
	*out << "  Parameter " << i << " = " << (*p)[i] << std::endl;
      *out << "END QCAD Solver Parameters" << std::endl;
    }
  }
   
  if( problemNameBase == "Poisson" ) {
      if(bVerbose) *out << "QCAD Solve: Simple Poisson solve" << std::endl;
      QCAD::SolveModel(getSubSolver("Poisson"));
      eigenvalueResponses.resize(0); // no eigenvalues in the Poisson problem
  }

  else if( problemNameBase == "Schrodinger" ) {
      if(bVerbose) *out << "QCAD Solve: Simple Schrodinger solve" << std::endl;
      Teuchos::RCP<Albany::EigendataStruct> eigenData = Teuchos::null;
      Teuchos::RCP<Albany::EigendataStruct> eigenDataNull = Teuchos::null;
      QCAD::SolveModel(getSubSolver("Schrodinger"), eigenDataNull, eigenData);
      eigenvalueResponses = *(eigenData->eigenvalueRe); // copy eigenvalues to member variable
      for(std::size_t i=0; i<eigenvalueResponses.size(); ++i) eigenvalueResponses[i] *= -1; //apply minus sign (b/c of eigenval convention)

      // Invoke final observer to output evecs and solution
      Teuchos::RCP<Epetra_Vector> solnVec = getSubSolver("Schrodinger").responses_out->get_g(1); //get the *first* response vector (solution)
      final_obs->observeSolution(*solnVec, "ZeroSolution", eigenData, 0.0);
  }

  else if( problemNameBase == "Poisson Schrodinger" )
    evalPoissonSchrodingerModel(inArgs, outArgs, eigenvalueResponses);

  else if( problemNameBase == "Schrodinger CI" )
    evalCIModel(inArgs, outArgs, eigenvalueResponses);

  else if( problemNameBase == "Poisson Schrodinger CI" )
    evalPoissonCIModel(inArgs, outArgs, eigenvalueResponses);

  if(num_g > 0) {
    // update main solver's responses using sub-solver response values
    Teuchos::RCP<Epetra_Vector> g = outArgs.get_g(0); //only use *first* response vector
    Teuchos::RCP<Epetra_MultiVector> dgdp = Teuchos::null;
    
    if(num_p > 0 && !outArgs.supports(OUT_ARG_DgDp, 0, 0).none()) 
      dgdp = outArgs.get_DgDp(0,0).getMultiVector();
    
    int offset = 0;
    std::vector<Teuchos::RCP<QCAD::SolverResponseFn> >::const_iterator rit;
    
    for(rit = responseFns.begin(); rit != responseFns.end(); rit++) {
      (*rit)->fillSolverResponses( *g, dgdp, offset, subSolvers, paramFnVecs, bSupportDpDg, eigenvalueResponses);
      offset += (*rit)->getNumDoubles();
    }
    
    if(bVerbose) {
      *out << "BEGIN QCAD Solver Responses:" << std::endl;
      for(int i=0; i< g->MyLength(); i++)
	*out << "  Response " << i << " = " << (*g)[i] << std::endl;
      *out << "END QCAD Solver Responses" << std::endl;
      
      //Seems to be a problem with print and MPI calls...
      /*if(!outArgs.supports(OUT_ARG_DgDp, 0, 0).none()) {
       *out << "BEGIN QCAD Solver Sensitivities:" << std::endl;
       dgdp->Print(*out);
       *out << "END QCAD Solver Sensitivities" << std::endl;
       }*/
    }
  }
}


void 
QCAD::Solver::evalPoissonSchrodingerModel(const InArgs& inArgs,
					  const OutArgs& outArgs,
					  std::vector<double>& eigenvalueResponses) const
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  //state variables
  Albany::StateArrays* pStatesToPass = NULL;
  Albany::StateArrays* pStatesToLoop = NULL; 
  Teuchos::RCP<Albany::EigendataStruct> eigenDataToPass = Teuchos::null;
  Teuchos::RCP<Albany::EigendataStruct> eigenDataNull = Teuchos::null;

  //Field Containers to store states used in Poisson-Schrodinger loop
  std::vector<Intrepid::FieldContainer<RealType> > prevElectricPotential;
  std::vector<Intrepid::FieldContainer<RealType> > tmpContainer;
  
  if(bVerbose) *out << "QCAD Solve: Initial Poisson solve (no quantum region) " << std::endl;
  QCAD::SolveModel(getSubSolver("InitPoisson"), pStatesToPass, pStatesToLoop);
    
  if(bVerbose) *out << "QCAD Solve: Beginning Poisson-Schrodinger solve loop" << std::endl;
  bool bConverged = false; 
  std::size_t iter = 0;
  double newShift;
    
  Teuchos::RCP<Teuchos::ParameterList> eigList; //used to hold memory I think - maybe unneeded?

  while(!bConverged && iter < maxIter)
  {
    iter++;

    if (iter == 1) 
      newShift = QCAD::GetEigensolverShift(getSubSolver("InitPoisson"), shiftPercentBelowMin);
    else
      newShift = QCAD::GetEigensolverShift(getSubSolver("Poisson"), shiftPercentBelowMin);
    QCAD::ResetEigensolverShift(getSubSolver("Schrodinger").model, newShift, eigList);

    // Schrodinger Solve -> eigenstates
    if(bVerbose) *out << "QCAD Solve: Schrodinger iteration " << iter << std::endl;
    QCAD::SolveModel(getSubSolver("Schrodinger"), pStatesToLoop, pStatesToPass,
	       eigenDataNull, eigenDataToPass);

    // Save solution for predictor-corrector outer iterations      
    QCAD::CopyStateToContainer(*pStatesToLoop, "PS Saved Solution", tmpContainer);
    QCAD::CopyContainerToState(tmpContainer, *pStatesToPass, "PS Previous Poisson Potential");

    // Poisson Solve
    if(bVerbose) *out << "QCAD Solve: Poisson iteration " << iter << std::endl;
    QCAD::SolveModel(getSubSolver("Poisson"), pStatesToPass, pStatesToLoop,
	       eigenDataToPass, eigenDataNull);

    eigenDataNull = Teuchos::null;

    if(iter > 1) {
      double local_maxDiff = QCAD::getMaxDifference(*pStatesToLoop, prevElectricPotential, "PS Saved Electric Potential");
      double global_maxDiff;
      solverComm->MaxAll(&local_maxDiff, &global_maxDiff, 1);
      bConverged = (global_maxDiff < CONVERGE_TOL);
      if(bVerbose) *out << "QCAD Solve: Electric Potential max diff=" 
			<< global_maxDiff << " (tol=" << CONVERGE_TOL << ")" << std::endl;
    }
      
    QCAD::CopyStateToContainer(*pStatesToLoop, "PS Saved Electric Potential", prevElectricPotential);
  } 

  eigenvalueResponses = *(eigenDataToPass->eigenvalueRe); // copy eigenvalues to member variable
  for(std::size_t i=0; i<eigenvalueResponses.size(); ++i) eigenvalueResponses[i] *= -1; //apply minus sign (b/c of eigenval convention)

  if(!bUseIntegratedPS) {
    if(bConverged) {
      // LATER: perhaps run a separate Poisson solve (as above) but have it compute all the responses we want
      //  (and don't have it compute them in the in-loop call above).
      
      // TODO: why is this here??  for iQCAD parsing??
      //Write parameters and responses of final Poisson solve
      // Don't worry about sensitivities yet - just output vectors
      
      const QCAD::SolverSubSolver& ss = getSubSolver("Poisson");
      int poisson_num_p = ss.params_in->Np();     // Number of *vectors* of parameters
      int poisson_num_g = ss.responses_out->Ng(); // Number of *vectors* of responses
      
      for (int i=0; i<poisson_num_p; i++)
	ss.params_in->get_p(i)->Print(*out << "\nPoisson Parameter vector " << i << ":\n");
      
      for (int i=0; i<poisson_num_g-1; i++) {
	Teuchos::RCP<Epetra_Vector> g = ss.responses_out->get_g(i);
	bool is_scalar = true;

	if (ss.app != Teuchos::null)
	  is_scalar = ss.app->getResponse(i)->isScalarResponse();

	if (is_scalar) {
	  g->Print(*out << "\nPoisson Response vector " << i << ":\n");
	  *out << "\n";  //add blank line after vector is printed - needed for proper post-processing
	  // see Main_Solve.cpp for how to print sensitivities here
	}
      }

      // Invoke final observer to output evecs (MB densities) and Poisson solution
      Teuchos::RCP<Epetra_Vector> solnVec = getSubSolver("Poisson").responses_out->get_g(1); //get the *first* response vector (solution)
      final_obs->observeSolution(*solnVec, "Potential", eigenDataToPass, 0.0);
    }


    if(bVerbose) {
      if(bConverged)
	*out << "QCAD Solve: Converged Poisson-Schrodinger solve loop after " << iter << " iterations." << std::endl;
      else
	*out << "QCAD Solve: Maximum iterations (" << maxIter << ") reached." << std::endl;
    }
  }

  else { // perform integrated poisson-schrodinger solve now

    if(bVerbose) {
      if(bConverged) *out << "QCAD Solve: Converged Poisson-Schrodinger solve loop after " << iter << " iterations." << std::endl;
      else           *out << "QCAD Solve: Integrated Poisson-Schrodinger solve started after " << iter << " S-P iterations."  << std::endl;
    }

    // Run Integrated solver always, even if iterative solver has converged to it's specified tolerance

    // get combined S-P map -- utilize a dummy CoupledPoissonSchrodinger object to do this for us...
    const Teuchos::RCP<Teuchos::ParameterList>& ps_paramList = subProblemAppParams.find("PoissonSchrodinger")->second;

    const Teuchos::RCP<QCAD::CoupledPoissonSchrodinger> ps_dummy = 
    	Teuchos::rcp(new QCAD::CoupledPoissonSchrodinger( ps_paramList, solverComm, Teuchos::null));

    Teuchos::RCP<const Epetra_Map> combinedMap = ps_dummy->get_x_map();

    // build initial guess from poisson potential, eigenvectors, and eigenvalues
    Teuchos::RCP<Epetra_Vector> initial_guess = Teuchos::rcp(new Epetra_Vector(*combinedMap));
    Teuchos::RCP<Epetra_Vector> initial_poisson, initial_evals;
    Teuchos::RCP<Epetra_MultiVector> initial_schrodinger;
    ps_dummy->separateCombinedVector(initial_guess, initial_poisson, initial_schrodinger, initial_evals);
    
    Teuchos::RCP<Epetra_Vector> poisson_soln = getSubSolver("Poisson").responses_out->get_g(1); // get the solution vector 
    initial_poisson->Scale(1.0, *poisson_soln); // initial_poisson = poisson_soln


    // Exporter to non-overlapped data (eigenvectors in EigendataInfo are stored in overlapped distribution)
    Teuchos::RCP<Albany::AbstractDiscretization> disc = getSubSolver("Poisson").app->getDiscretization();
    Teuchos::RCP<const Epetra_Map> disc_map = disc->getMap();
    Teuchos::RCP<const Epetra_Map> disc_overlap_map = disc->getOverlapMap();
    Teuchos::RCP<Epetra_Export> overlap_exporter = Teuchos::rcp(new Epetra_Export(*disc_overlap_map, *disc_map));

    int nEigenvals = initial_schrodinger->NumVectors();
    for(int k=0; k < nEigenvals; k++) 
    	(*initial_schrodinger)(k)->Export( *((*(eigenDataToPass->eigenvectorRe))(k)), *overlap_exporter, Insert);

    std::vector<int> myGlobalEls( initial_evals->MyLength() );
    initial_evals->Map().MyGlobalElements(&myGlobalEls[0]);
    for(std::size_t k=0; k < myGlobalEls.size(); k++) {
    	(*initial_evals)[k] = (*(eigenDataToPass->eigenvalueRe))[ myGlobalEls[k] ];

    	//DEBUG
    	double ignored_norm;
    	(*(eigenDataToPass->eigenvectorIm))(myGlobalEls[k])->Norm2(&ignored_norm);
    	std::cout << "DEBUG: ignored imaginary part with norm = " << ignored_norm << std::endl;
    	//DEBUG
    }
    	    
    QCAD::SolverSubSolver coupledPS_Solver = CreateSubSolver(ps_paramList, *solverComm, initial_guess);
    QCAD::SolveModel(coupledPS_Solver);

    //Pull eigenvalues out of solution into eigenvalueResponses
    Teuchos::RCP<Epetra_Vector> solutionVec = outArgs.get_g(1); // 2nd response vector == solution?? -- really should be *last* response vector
    Teuchos::RCP<Epetra_Vector> soln_poisson, soln_evals;
    Teuchos::RCP<Epetra_MultiVector> soln_schrodinger;
    ps_dummy->separateCombinedVector(solutionVec, soln_poisson, soln_schrodinger, soln_evals);

    Epetra_LocalMap local_eigenval_map(nEigenvals, 0, *solverComm);
    Epetra_Import eigenval_importer(local_eigenval_map, soln_evals->Map());

    Teuchos::RCP<Epetra_Vector> eigenvals =  Teuchos::rcp(new Epetra_Vector(local_eigenval_map));
    eigenvals->Import(*soln_evals, eigenval_importer, Insert);
    eigenvalueResponses = std::vector<double>(&(*eigenvals)[0], &(*eigenvals)[0] + nEigenvals);
    for(std::size_t i=0; i<eigenvalueResponses.size(); ++i) eigenvalueResponses[i] *= -1; //apply minus sign (b/c of eigenval convention)
  }
}



void 
QCAD::Solver::evalCIModel(const InArgs& inArgs,
			  const OutArgs& outArgs,
			  std::vector<double>& eigenvalueResponses) const
{
#ifdef ALBANY_CI
  // const double CONVERGE_TOL = 1e-5;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  //state variables
  Albany::StateArrays* pStatesToPass = NULL;
  Albany::StateArrays* pStatesToLoop = NULL; 
  Teuchos::RCP<Albany::EigendataStruct> eigenDataToPass = Teuchos::null;
  Teuchos::RCP<Albany::EigendataStruct> eigenDataNull = Teuchos::null;

  if(bVerbose) *out << "QCAD Solve: CI solve" << std::endl;

  Teuchos::RCP<Teuchos::ParameterList> eigList; //used to hold memory I think - maybe unneeded?
  int n1PperBlock = nEigenvectors;

  //Memory for CI Matrices
  //  - H1P matrix blocks (generate 2 blocks (up & down), each nEvecs x nEvecs)
  std::vector<Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > > blocks1P;
  Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > blockU,blockD;
  blockU = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock, n1PperBlock));
  blockD = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock, n1PperBlock));
  blocks1P.push_back(blockU); blocks1P.push_back(blockD);

  //  - H2P matrix blocks (generate 4 blocks, each nEvecs x nEvecs x nEvecs x nEvecs)
  std::vector<Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > > blocks2P;
  Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > blockUU, blockUD, blockDU, blockDD;
  blockUU = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock,n1PperBlock,n1PperBlock,n1PperBlock));
  blockUD = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock,n1PperBlock,n1PperBlock,n1PperBlock));
  blockDU = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock,n1PperBlock,n1PperBlock,n1PperBlock));
  blockDD = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock,n1PperBlock,n1PperBlock,n1PperBlock));
  blocks2P.push_back(blockUU);  blocks2P.push_back(blockUD);
  blocks2P.push_back(blockDU);  blocks2P.push_back(blockDD);

  // 1P basis - assume Sz-symmetry to N up sptl wfns, N dn sptl wfns
  Teuchos::RCP<AlbanyCI::SingleParticleBasis> basis1P = Teuchos::rcp(new AlbanyCI::SingleParticleBasis);
  basis1P->init(n1PperBlock /* nUpFns */ , n1PperBlock /* nDnFns */, true /*bMxsHaveBlkStructure*/ ); 
  //basis1P->print(out); //DEBUG


  // Setup CI Solver
  std::vector<int> nParticlesInSubbases(1);
  AlbanyCI::symmetrySet symmetries;
  AlbanyCI::SymmetryFilters symmetryFilters;

  Teuchos::RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(Albany::getMpiCommFromEpetraComm( *solverComm));
  Teuchos::RCP<Teuchos::ParameterList> MyPL = Teuchos::rcp( new Teuchos::ParameterList );

  //Set blockSize and numBlocks based on number of eigenvalues computed and the size of the problem
  int numEvals=nEigenvectors;
  int blockSize=nEigenvectors + 1; //better guess? - base on size of matrix?
  int numBlocks=8;  //better guess?

  int maxRestarts=100;
  double tol = 1e-7;

  //Defaults
  MyPL->set("Num Excitations", 0);
  MyPL->set("Num Subbases", 1);
  MyPL->set("Subbasis Particles 0", 0);

  /*// Block-diagonalize Hamiltonian based on Sz-symmetry (only)
  MyPL->set("Num Symmetries", 1);
  MyPL->set("Symmetry 0", "Sz"); */

  // Block-diag Hamiltonian by S2 and Sz symmetries
  MyPL->set("Num Symmetries", 2);
  MyPL->set("Symmetry 0", "Sz");
  MyPL->set("Symmetry 1", "S2");

  MyPL->set("Num Symmetry Filters", 0);



  Teuchos::ParameterList& AnasaziList = MyPL->sublist("Anasazi");
  std::string which("SR");
  AnasaziList.set( "Which", which );
  AnasaziList.set( "Num Eigenvalues", numEvals );
  AnasaziList.set( "Block Size", blockSize );
  AnasaziList.set( "Num Blocks", numBlocks );
  AnasaziList.set( "Maximum Restarts", maxRestarts );
  AnasaziList.set( "Convergence Tolerance", tol );

  MyPL->set("Num Excitations", nCIExcitations);
  MyPL->set("Subbasis Particles 0", nCIParticles);	  

  // Schrodinger Solve -> eigenstates
  if(bVerbose) *out << "QCAD Solve: Schrodinger solve" << std::endl;

  QCAD::SolveModel(getSubSolver("Schrodinger"), pStatesToLoop, pStatesToPass,
		   eigenDataNull, eigenDataToPass);
     
  // Construct CI matrices: (see comments in evalPoissonCIModel)
  // TODO: perhaps consolidate this code with that in evalPoissonCIModel to avoid duplication
    
  // transfer responses to H1P matrix (2 blocks (up & down), each nEvecs x nEvecs)
  int rIndx = 0; // offset to the responses corresponding to delta_ij values == 0 by construction
  for(int i=0; i<nEigenvectors; i++) {
    blockU->el(i,i) = -(*(eigenDataToPass->eigenvalueRe))[i]; // minus (-) sign b/c of 
    blockD->el(i,i) = -(*(eigenDataToPass->eigenvalueRe))[i]; //  eigenvalue convention
    *out << "DEBUG CI 1P Block El (" <<i<<","<<i<<") = " << -(*(eigenDataToPass->eigenvalueRe))[i] << std::endl;
    
    for(int j=i+1; j<nEigenvectors; j++) {
      blockU->el(i,j) = 0; blockU->el(j,i) = 0;
      blockD->el(i,j) = 0; blockD->el(j,i) = 0;
    }
  }
    
  //DEBUG
  //*out << "DEBUG: g vector:" << std::endl;
  //for(int i=0; i< g->MyLength(); i++) *out << "  g[" << i << "] = " << (*g)[i] << std::endl;
          
  Teuchos::RCP<AlbanyCI::BlockTensor<AlbanyCI::dcmplx> > mx1P =
    Teuchos::rcp(new AlbanyCI::BlockTensor<AlbanyCI::dcmplx>(basis1P, blocks1P, 1));
  //*out << std::endl << "DEBUG CI mx1P:"; mx1P->print(out); //DEBUG
    
                
  // fill in mx2P (4 blocks, each n1PperBlock x n1PperBlock x n1PperBlock x n1PperBlock )
  for(int i2=0; i2<nEigenvectors; i2++) {
    for(int i4=i2; i4<nEigenvectors; i4++) {
      
      // Coulomb Poisson Solve - get coulomb els in reponse vector
      if(bVerbose) *out << "QCAD Solve: Coulomb " << i2 << "," << i4 << " Poisson" << std::endl;
      SetCoulombParams( getSubSolver("CoulombPoisson").params_in, i2,i4 ); 
      QCAD::SolveModel(getSubSolver("CoulombPoisson"), pStatesToPass, pStatesToLoop,
			       eigenDataToPass, eigenDataNull);
    	  
      // transfer responses to H2P matrix blocks
      Teuchos::RCP<Epetra_Vector> g_reSrc =
	getSubSolver("CoulombPoisson").responses_out->get_g(0); //only use *first* response vector    

      //DEBUG
      *out << "DEBUG: g_reSrc vector:" << std::endl;
      for(int i=0; i< g_reSrc->MyLength(); i++) *out << "  g_reSrc[" << i << "] = " << (*g_reSrc)[i] << std::endl;	      
	      
      // Coulomb Poisson Solve - get coulomb els in reponse vector
      if(bVerbose) *out << "QCAD Solve: Imaginary Coulomb " << i2 << "," << i4 << " Poisson" << std::endl;
      SetCoulombParams( getSubSolver("CoulombPoissonIm").params_in, i2,i4 ); 
      QCAD::SolveModel(getSubSolver("CoulombPoissonIm"), pStatesToPass, pStatesToLoop,
		       eigenDataToPass, eigenDataNull);
    	  
      // transfer responses to H2P matrix blocks
      Teuchos::RCP<Epetra_Vector> g_imSrc =
	getSubSolver("CoulombPoissonIm").responses_out->get_g(0); //only use *first* response vector    
	      
      //DEBUG
      *out << "DEBUG: g_imSrc vector:" << std::endl;
      for(int i=0; i< g_imSrc->MyLength(); i++) *out << "  g_imSrc[" << i << "] = " << (*g_imSrc)[i] << std::endl;
	      
      rIndx = 0 ;  // offset to the responses corresponding to Coulomb_ij values == 0 by construction
      for(int i1=0; i1<nEigenvectors; i1++) {
	for(int i3=i1; i3<nEigenvectors; i3++) {
	  assert(rIndx < g_reSrc->MyLength()); //make sure g-vector is long enough
	  double c_reSrc_re = -(*g_reSrc)[rIndx];
	  double c_reSrc_im = -(*g_reSrc)[rIndx+1]; // rIndx + 1 == imag part
	  double c_imSrc_re = -(*g_imSrc)[rIndx];
	  double c_imSrc_im = -(*g_imSrc)[rIndx+1]; // rIndx + 1 == imag part
		  
	  //Coulomb integral of interest (see above)
	  double c_re = c_reSrc_re - c_imSrc_im;
	  double c_im = c_reSrc_im + c_imSrc_re;
		  
		  
	  *out << "DEBUG CI 2P Block El (" <<i1<<","<<i2<<","<<i3<<","<<i4<<") = " << c_re << " + i*" << c_im << std::endl;
		  
	  // Only use REAL parts here since we don't have complex support yet
	  //  (Tpetra doesn't work).  Use c_re + i*c_im or conjugate where necessary.
	  blockUU->el(i1,i2,i3,i4) = c_re;
	  blockUU->el(i3,i2,i1,i4) = c_re;
	  blockUU->el(i1,i4,i3,i2) = c_re;
	  blockUU->el(i3,i4,i1,i2) = c_re;
	  
	  blockUD->el(i1,i2,i3,i4) = c_re;
	  blockUD->el(i3,i2,i1,i4) = c_re;
	  blockUD->el(i1,i4,i3,i2) = c_re;
	  blockUD->el(i3,i4,i1,i2) = c_re;
		  
	  blockDU->el(i1,i2,i3,i4) = c_re;
	  blockDU->el(i3,i2,i1,i4) = c_re;
	  blockDU->el(i1,i4,i3,i2) = c_re;
	  blockDU->el(i3,i4,i1,i2) = c_re;
	  
	  blockDD->el(i1,i2,i3,i4) = c_re;
	  blockDD->el(i3,i2,i1,i4) = c_re;
	  blockDD->el(i1,i4,i3,i2) = c_re;
	  blockDD->el(i3,i4,i1,i2) = c_re;
		  
	  rIndx += 2;
	}
      }
    }
  }
          
  Teuchos::RCP<AlbanyCI::BlockTensor<AlbanyCI::dcmplx> > mx2P =
    Teuchos::rcp(new AlbanyCI::BlockTensor<AlbanyCI::dcmplx>(basis1P, blocks2P, 2));
  //*out << std::endl << "DEBUG CI mx2P:"; mx2P->print(out); //DEBUG
         
          
  //Now should have H1P and H2P - run CI:
  if(bVerbose) *out << "QCAD Solve: CI solve" << std::endl;
	  
  AlbanyCI::Solver solver;
  Teuchos::RCP<AlbanyCI::Solution> soln;
  soln = solver.solve(MyPL, mx1P, mx2P, tcomm, out); //Note: out cannot be null
  //*out << std::endl << "Solution:"; soln->print(out); //DEBUG

  eigenvalueResponses = soln->getEigenvalues(); // save CI eigenvalues in member variable for responses
	  
  // NOTE: We'd like to compute and output to exodus the many-body wavefunction density
  //  here, but I'm (erik) not sure how to do this best.  So I'm commenting out the 
  //  computation of the MB densties for now, and hopefully in the future we can add this
  //  exporting to exodus capability
  
  // Compute the total electron density for each eigenstate and overwrite the 
  //  eigenvector real part with this data
  std::vector<double> eigenvalues = soln->getEigenvalues();
  int nCIevals = eigenvalues.size();
  std::vector< std::vector< AlbanyCI::dcmplx > > mxPx;
  //Teuchos::RCP<AlbanyCI::Solution::Vector> ci_evec;
	  
  Teuchos::RCP<Epetra_MultiVector> mbStateDensities = 
    Teuchos::rcp( new Epetra_MultiVector(eigenDataToPass->eigenvectorRe->Map(), nCIevals, true )); //zero out
  eigenDataToPass->eigenvalueRe->resize(nCIevals);
  eigenDataToPass->eigenvalueIm->resize(nCIevals);
	  
  //int rank = tcomm->getRank();
  //std::cout << "DEBUG Rank " << rank << ": " << nCIevals << " evals" << std::endl;
  
  for(int k=0; k < nCIevals; k++) {
    soln->getEigenvectorPxMatrix(k, mxPx); // mxPx = n1P x n1P matrix of coeffs of 1P products
    (*(eigenDataToPass->eigenvalueRe))[k] = eigenvalues[k];
    (*(eigenDataToPass->eigenvalueIm))[k] = 0.0; //evals are real
	    
    //Note that CI's n1P is twice the number of eigenvalues in Albany eigendata due to spin degeneracy
    // and we must sum up and down parts [2*i and 2*i+1 ==> spatial evec i] -- LATER: get this info from soln?
    for(int i=0; i < n1PperBlock; i++) {
      const Epetra_Vector& vi_real = *((*(eigenDataToPass->eigenvectorRe))(i));
      const Epetra_Vector& vi_imag = *((*(eigenDataToPass->eigenvectorIm))(i));
	      
      for(int j=0; j < n1PperBlock; j++) {
	const Epetra_Vector& vj_real = *((*(eigenDataToPass->eigenvectorRe))(j));
	const Epetra_Vector& vj_imag = *((*(eigenDataToPass->eigenvectorIm))(j));
		
	(*mbStateDensities)(k)->Multiply( mxPx[i][j], vi_real, vj_real, 1.0); // mbDen(k) += mxPx_ij * elwise(Vi_r * Vj_r)
	(*mbStateDensities)(k)->Multiply( mxPx[i][j], vi_imag, vj_imag, 1.0); // mbDen(k) += mxPx_ij * elwise(Vi_i * Vj_i)
      }
    }
  }
    
  // Put densities into eigenDataToPass (evals already done above):
  //   (just for good measure duplicate in re and im multivecs so they're the same size - probably unecessary)
  eigenDataToPass->eigenvectorRe = mbStateDensities;
  eigenDataToPass->eigenvectorIm = mbStateDensities; 
  
  
  if(bVerbose) *out << "QCAD Solve: CI solve finished." << std::endl;


  // Invoke final observer to output evecs (MB densities) and solution
  Teuchos::RCP<Epetra_Vector> solnVec = getSubSolver("Schrodinger").responses_out->get_g(1); //get the *first* response vector (solution)
  final_obs->observeSolution(*solnVec, "ZeroSolution", eigenDataToPass, 0.0);


  /* TODO - what to put as final responses?
  if(bConverged) {
    // LATER: perhaps run a separate Poisson CI solve (as above) but have it compute all the responses we want
    //  (and don't have it compute them in the in-loop call above).

    //Write parameters and responses of final PoissonCI solve
    // Don't worry about sensitivities yet - just output vectors

    const QCAD::SolverSubSolver& ss = getSubSolver("CIPoisson");
    int cipoisson_num_p = ss.params_in->Np();     // Number of *vectors* of parameters
    int cipoisson_num_g = ss.responses_out->Ng(); // Number of *vectors* of responses

    for (int i=0; i<cipoisson_num_p; i++)
      ss.params_in->get_p(i)->Print(*out << "\nParameter vector " << i << ":\n");

    for (int i=0; i<cipoisson_num_g-1; i++) {
      Teuchos::RCP<Epetra_Vector> g = ss.responses_out->get_g(i);
      bool is_scalar = true;

      if (ss.app != Teuchos::null)
        is_scalar = ss.app->getResponse(i)->isScalarResponse();

      if (is_scalar) {
        g->Print(*out << "\nCIPoisson Response vector " << i << ":\n");
	*out << "\n";  //add blank line after vector is printed - needed for proper post-processing
	// see Main_Solve.cpp for how to print sensitivities here
      }
    }
  }
  */

#else
  
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       "Albany must be built with ALBANY_CI enabled in order to perform CI solutions." << std::endl);

#endif
}



void 
QCAD::Solver::evalPoissonCIModel(const InArgs& inArgs,
				 const OutArgs& outArgs,
				 std::vector<double>& eigenvalueResponses) const
{
#ifdef ALBANY_CI
  // const double CONVERGE_TOL = 1e-5;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  //state variables
  Albany::StateArrays* pStatesToPass = NULL;
  //Albany::StateArrays* pStatesFromDummy = NULL;
  Albany::StateArrays* pStatesToLoop = NULL; 
  Teuchos::RCP<Albany::EigendataStruct> eigenDataToPass = Teuchos::null;
  Teuchos::RCP<Albany::EigendataStruct> eigenDataNull = Teuchos::null;

  //Field Containers to store states used in Poisson-Schrodinger loop
  std::vector<Intrepid::FieldContainer<RealType> > prevElectricPotential;
  std::vector<Intrepid::FieldContainer<RealType> > tmpContainer;
 
  if(bVerbose) *out << "QCAD Solve: Initial Poisson solve (no quantum region) " << std::endl;
  QCAD::SolveModel(getSubSolver("InitPoisson"), pStatesToPass, pStatesToLoop);
    
  if(bVerbose) *out << "QCAD Solve: Beginning Poisson-CI solve loop" << std::endl;
  bool bConverged = false;
  bool bPoissonSchrodingerConverged = false;
  bool bRunCI = true;
  std::size_t iter = 0;
  double newShift;

  Teuchos::RCP<Teuchos::ParameterList> eigList; //used to hold memory I think - maybe unneeded?
  int n1PperBlock = nEigenvectors;

  //Memory for CI Matrices
  //  - H1P matrix blocks (generate 2 blocks (up & down), each nEvecs x nEvecs)
  std::vector<Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > > blocks1P;
  Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > blockU,blockD;
  blockU = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock, n1PperBlock));
  blockD = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock, n1PperBlock));
  blocks1P.push_back(blockU); blocks1P.push_back(blockD);

  //  - H2P matrix blocks (generate 4 blocks, each nEvecs x nEvecs x nEvecs x nEvecs)
  std::vector<Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > > blocks2P;
  Teuchos::RCP<AlbanyCI::Tensor<AlbanyCI::dcmplx> > blockUU, blockUD, blockDU, blockDD;
  blockUU = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock,n1PperBlock,n1PperBlock,n1PperBlock));
  blockUD = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock,n1PperBlock,n1PperBlock,n1PperBlock));
  blockDU = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock,n1PperBlock,n1PperBlock,n1PperBlock));
  blockDD = Teuchos::rcp(new AlbanyCI::Tensor<AlbanyCI::dcmplx>(n1PperBlock,n1PperBlock,n1PperBlock,n1PperBlock));
  blocks2P.push_back(blockUU);  blocks2P.push_back(blockUD);
  blocks2P.push_back(blockDU);  blocks2P.push_back(blockDD);

  // 1P basis - assume Sz-symmetry to N up sptl wfns, N dn sptl wfns
  Teuchos::RCP<AlbanyCI::SingleParticleBasis> basis1P = Teuchos::rcp(new AlbanyCI::SingleParticleBasis);
  basis1P->init(n1PperBlock /* nUpFns */ , n1PperBlock /* nDnFns */, true /*bMxsHaveBlkStructure*/ ); 
  //basis1P->print(out); //DEBUG


  // Setup CI Solver
  std::vector<int> nParticlesInSubbases(1);
  AlbanyCI::symmetrySet symmetries;
  AlbanyCI::SymmetryFilters symmetryFilters;

  Teuchos::RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(Albany::getMpiCommFromEpetraComm( *solverComm));
  Teuchos::RCP<Teuchos::ParameterList> MyPL = Teuchos::rcp( new Teuchos::ParameterList );

  //Set blockSize and numBlocks based on number of eigenvalues computed and the size of the problem
  int numEvals=nEigenvectors;
  int blockSize=nEigenvectors + 1; //better guess? - base on size of matrix?
  int numBlocks=8;  //better guess?

  int maxRestarts=100;
  double tol = 1e-8;

  //Defaults
  MyPL->set("Num Excitations", 0);
  MyPL->set("Num Subbases", 1);
  MyPL->set("Subbasis Particles 0", 0);
  MyPL->set("Num Symmetries", 1);
  MyPL->set("Symmetry 0", "Sz");
  MyPL->set("Num Symmetry Filters", 0);

  Teuchos::ParameterList& AnasaziList = MyPL->sublist("Anasazi");
  std::string which("SR");
  AnasaziList.set( "Which", which );
  AnasaziList.set( "Num Eigenvalues", numEvals );
  AnasaziList.set( "Block Size", blockSize );
  AnasaziList.set( "Num Blocks", numBlocks );
  AnasaziList.set( "Maximum Restarts", maxRestarts );
  AnasaziList.set( "Convergence Tolerance", tol );
    
  eigenvalueResponses.resize(0);

  //Loop: 
  // 1) converge Schrodinger-Poisson as in evalPoissonSchrodingerModel
  // 2) get the number electrons in the quantum regions
  // 3) loop with CI included

  while(!bConverged && iter < maxIter) 
  {
    iter++;
 
    if (iter == 1) 
      newShift = QCAD::GetEigensolverShift(getSubSolver("InitPoisson"), shiftPercentBelowMin);
    else
      newShift = QCAD::GetEigensolverShift(getSubSolver("Poisson"), shiftPercentBelowMin);
    QCAD::ResetEigensolverShift(getSubSolver("Schrodinger").model, newShift, eigList);

    // Schrodinger Solve -> eigenstates
    if(bVerbose) *out << "QCAD Solve: Schrodinger iteration " << iter << std::endl;
    QCAD::SolveModel(getSubSolver("Schrodinger"), pStatesToLoop, pStatesToPass,
	       eigenDataNull, eigenDataToPass);
     
    // Save solution for predictory-corrector outer iterations
    QCAD::CopyStateToContainer(*pStatesToLoop, "PS Saved Solution", tmpContainer);
    QCAD::CopyContainerToState(tmpContainer, *pStatesToPass, "PS Previous Poisson Potential");
      
    if(bPoissonSchrodingerConverged) {

      if(bRunCI) {
    
          // Construct CI matrices:
          // For N eigenvectors:
          //  1) a NxN matrix of the single particle hamiltonian: H1P. Derivation:
          //    H1P = diag(E) - delta, where delta_ij = int( [i(r)j(r)] F(r) dr)
          //    - no actual poisson solve needed, but need framework to integrate?
          //    - could we save a state containing weights and then integrate outside of NOX?
          //
          //  2) a matrix of all pair integrals.  Derivation:
          //    <12|1/r|34> = int( 1(r1) 2(r2) 1/(r1-r2) 3(r1) 4(r2) dr1 dr2)
          //                = int( 1(r1) 3(r1) [ int( 2(r2) 1/(r1-r2) 4(r2) dr2) ] dr1 )
          //                = int( 1(r1) 3(r1) [ soln of Poisson, rho(r1), with src = 2(r) 4(r) ] dr1 )
          //    - so use dummy poisson solve which has i(r)j(r) as only RHS term, for each pair <ij>,
          //       and as output of each Solve integrate wrt each other potential pair 1(r) 3(r)
          //       to generate all the elements.
          
          // What is F(r)?
          // evecs given are evecs of H = T + V + Vxc where V includes coulomb interaction with all charge
          //                            = T + V(src = other CI electrons) + V(src = classical charge) + Vxc
          // and we want matrix of H1P = T + V(src = classical charge)
          // So <i|H1P|j> = <i|H - V(src = other CI electrons) - Vxc|j> = E_i * Delta(i,j) - <i| V(src = other CI) + Vxc |j>
          //                = E_i * Delta(i,j) - delta_ij
          // and F(r) = [soln of Poisson, rho(r), with src = previous soln RHS restricted to quantum region] + Vxc(r)
          //     and the charge of just the quantum region is just determined by the eigenvectors/values, sum(state density * occupation)
          
          // Need to call Poisson in 
          // 1) "compute_delta" mode --> g[i*N+j] == delta_ij  (so N^2 responses)
          //    - set poisson source param specifying which reastricts RHS to quantum region
          //    - set poisson source param specifying quantum region charge == input from state (eigenstates & energies now - MB?)
          //    - solve with fixed charge in quantum region -- subregion of full coupled Poisson solve == source 
          //    - set responses to N^2 new "F(r)" responses which take "Weight State Name 1" and "2" params and integrate F(r) wrt them
          // 2) "compute_Coulomb" (i2,i4) mode --> g[i1*N+i3] == <i1,i2|1/r|i3,i4> (so N^2 responses)
          //    - set poisson source param which reastricts RHS to quantum region
          //    - set poisson source param specifying quantum region charge == product of eigenvectors & give i2,i4 indices
          //    - set responses to N^2 FieldIntegrals - ADD "Weight State Name 1" and "2" to possible params -- i1,i3
          // in both modes keep dbc's as they are - just change RHS of poisson.
    
          // Poisson Solve without any source charge: this gives terms that are due to environment charges that
          //   occur due to boundary conditions (e.g. charge on surface of conductors due to DBCs) that we must subtract
          //   from terms below to get effect of *just* the quantum electron charges and their image charges. 
          if(bVerbose) *out << "QCAD Solve: No-charge Poisson iteration " << iter << std::endl;
          QCAD::SolveModel(getSubSolver("NoChargePoisson"), pStatesToPass, pStatesToLoop,
			   eigenDataToPass, eigenDataNull);
          Teuchos::RCP<Epetra_Vector> g_noCharge =
	    getSubSolver("NoChargePoisson").responses_out->get_g(0); //only use *first* response vector    
    
          
          // Delta Poisson Solve - get delta_ij in reponse vector
          if(bVerbose) *out << "QCAD Solve: Delta Poisson iteration " << iter << std::endl;
          QCAD::SolveModel(getSubSolver("DeltaPoisson"), pStatesToPass, pStatesToLoop,
			   eigenDataToPass, eigenDataNull);
    
          // transfer responses to H1P matrix (2 blocks (up & down), each nEvecs x nEvecs)
          Teuchos::RCP<Epetra_Vector> g =
	    getSubSolver("DeltaPoisson").responses_out->get_g(0); //only use *first* response vector    
          int rIndx = 0; // offset to the responses corresponding to delta_ij values == 0 by construction
          for(int i=0; i<nEigenvectors; i++) {
	    assert(rIndx < g->MyLength()); //make sure g-vector is long enough
	    double delta_re = -( (*g)[rIndx] - (*g_noCharge)[rIndx] );       //Minus sign used because we use electric potential
	    double delta_im = -( (*g)[rIndx+1] - (*g_noCharge)[rIndx+1] );   // in delta calcs, and e- sees negated potential
	    blockU->el(i,i) = -(*(eigenDataToPass->eigenvalueRe))[i] - delta_re; // first minus (-) sign b/c of 
	    blockD->el(i,i) = -(*(eigenDataToPass->eigenvalueRe))[i] - delta_re; //  eigenvalue convention
	    *out << "DEBUG CI 1P Block El (" <<i<<","<<i<<") = " << -(*(eigenDataToPass->eigenvalueRe))[i] << " - " 
		 << "(" << delta_re << " + i*" << delta_im << ")" << std::endl;
	    rIndx += 2;
    
	    for(int j=i+1; j<nEigenvectors; j++) {
	      assert(rIndx < g->MyLength()); //make sure g-vector is long enough
	      delta_re = -((*g)[rIndx] - (*g_noCharge)[rIndx]);
	      delta_im = -((*g)[rIndx+1] - (*g_noCharge)[rIndx+1]);
	      blockU->el(i,j) = -delta_re; blockU->el(j,i) = -delta_re;
	      blockD->el(i,j) = -delta_re; blockD->el(j,i) = -delta_re;
	      *out << "DEBUG CI 1P Block El (" <<i<<","<<j<<") = (" << delta_re << " + i*" << delta_im << ")" << std::endl;
	      rIndx += 2;
	    }
          }
    
          //DEBUG
          //*out << "DEBUG: g vector:" << std::endl;
          //for(int i=0; i< g->MyLength(); i++) *out << "  g[" << i << "] = " << (*g)[i] << std::endl;
          
          Teuchos::RCP<AlbanyCI::BlockTensor<AlbanyCI::dcmplx> > mx1P =
	    Teuchos::rcp(new AlbanyCI::BlockTensor<AlbanyCI::dcmplx>(basis1P, blocks1P, 1));
          //*out << std::endl << "DEBUG CI mx1P:"; mx1P->print(out); //DEBUG
    
                
          // fill in mx2P (4 blocks, each n1PperBlock x n1PperBlock x n1PperBlock x n1PperBlock )
          for(int i2=0; i2<nEigenvectors; i2++) {
	    for(int i4=i2; i4<nEigenvectors; i4++) {
    	  
	      // Coulomb Poisson Solve - get coulomb els in reponse vector
	      if(bVerbose) *out << "QCAD Solve: Coulomb " << i2 << "," << i4 << " Poisson iteration " << iter << std::endl;
	      SetCoulombParams( getSubSolver("CoulombPoisson").params_in, i2,i4 ); 
	      QCAD::SolveModel(getSubSolver("CoulombPoisson"), pStatesToPass, pStatesToLoop,
			       eigenDataToPass, eigenDataNull);
    	  
	      // transfer responses to H2P matrix blocks
	      Teuchos::RCP<Epetra_Vector> g_reSrc =
		getSubSolver("CoulombPoisson").responses_out->get_g(0); //only use *first* response vector    
	      
	      
	      // Coulomb Poisson Solve - get coulomb els in reponse vector
	      if(bVerbose) *out << "QCAD Solve: Imaginary Coulomb " << i2 << "," << i4 << " Poisson iteration " << iter << std::endl;
	      SetCoulombParams( getSubSolver("CoulombPoissonIm").params_in, i2,i4 ); 
	      QCAD::SolveModel(getSubSolver("CoulombPoissonIm"), pStatesToPass, pStatesToLoop,
			       eigenDataToPass, eigenDataNull);
    	  
	      // transfer responses to H2P matrix blocks
	      Teuchos::RCP<Epetra_Vector> g_imSrc =
		getSubSolver("CoulombPoissonIm").responses_out->get_g(0); //only use *first* response vector    
	      
	      //DEBUG
	      //*out << "DEBUG: g vector:" << std::endl;
	      //for(int i=0; i< g->MyLength(); i++) *out << "  g[" << i << "] = " << (*g)[i] << std::endl;
	      
	      rIndx = 0 ;  // offset to the responses corresponding to Coulomb_ij values == 0 by construction
	      for(int i1=0; i1<nEigenvectors; i1++) {
		for(int i3=i1; i3<nEigenvectors; i3++) {
		  assert(rIndx < g->MyLength()); //make sure g-vector is long enough
		  double c_reSrc_re = -((*g_reSrc)[rIndx] - (*g_noCharge)[rIndx]);  
		  double c_reSrc_im = -((*g_reSrc)[rIndx+1] - (*g_noCharge)[rIndx+1]); // rIndx + 1 == imag part
		  double c_imSrc_re = -((*g_imSrc)[rIndx] - (*g_noCharge)[rIndx]);
		  double c_imSrc_im = -((*g_imSrc)[rIndx+1] - (*g_noCharge)[rIndx+1]); // rIndx + 1 == imag part
		  
		  //Coulomb integral of interest (see above)
		  double c_re = c_reSrc_re - c_imSrc_im;
		  double c_im = c_reSrc_im + c_imSrc_re;
		  
		  
		  *out << "DEBUG CI 2P Block El (" <<i1<<","<<i2<<","<<i3<<","<<i4<<") = " << c_re << " + i*" << c_im << std::endl;
		  
		  // Only use REAL parts here since we don't have complex support yet
		  //  (Tpetra doesn't work).  Use c_re + i*c_im or conjugate where necessary.
		  blockUU->el(i1,i2,i3,i4) = c_re;
		  blockUU->el(i3,i2,i1,i4) = c_re;
		  blockUU->el(i1,i4,i3,i2) = c_re;
		  blockUU->el(i3,i4,i1,i2) = c_re;
		  
		  blockUD->el(i1,i2,i3,i4) = c_re;
		  blockUD->el(i3,i2,i1,i4) = c_re;
		  blockUD->el(i1,i4,i3,i2) = c_re;
		  blockUD->el(i3,i4,i1,i2) = c_re;
		  
		  blockDU->el(i1,i2,i3,i4) = c_re;
		  blockDU->el(i3,i2,i1,i4) = c_re;
		  blockDU->el(i1,i4,i3,i2) = c_re;
		  blockDU->el(i3,i4,i1,i2) = c_re;
		  
		  blockDD->el(i1,i2,i3,i4) = c_re;
		  blockDD->el(i3,i2,i1,i4) = c_re;
		  blockDD->el(i1,i4,i3,i2) = c_re;
		  blockDD->el(i3,i4,i1,i2) = c_re;
		  
		  rIndx += 2;
		}
	      }
	    }
          }
          
          Teuchos::RCP<AlbanyCI::BlockTensor<AlbanyCI::dcmplx> > mx2P =
	    Teuchos::rcp(new AlbanyCI::BlockTensor<AlbanyCI::dcmplx>(basis1P, blocks2P, 2));
          //*out << std::endl << "DEBUG CI mx2P:"; mx2P->print(out); //DEBUG
         
          
          //Now should have H1P and H2P - run CI:
          if(bVerbose) *out << "QCAD Solve: CI solve" << std::endl;
	  
          AlbanyCI::Solver solver;
          Teuchos::RCP<AlbanyCI::Solution> soln;
          soln = solver.solve(MyPL, mx1P, mx2P, tcomm, out); //Note: out cannot be null
          //*out << std::endl << "Solution:"; soln->print(out); //DEBUG

	  eigenvalueResponses = soln->getEigenvalues(); // save CI eigenvalues in member variable for responses
	  
          // Compute the total electron density for each eigenstate and overwrite the 
          //  eigenvector real part with this data. (Expected by CIPoisson sub-solver)
          std::vector<double> eigenvalues = soln->getEigenvalues();
          int nCIevals = eigenvalues.size();
          std::vector< std::vector< AlbanyCI::dcmplx > > mxPx;
          //Teuchos::RCP<AlbanyCI::Solution::Vector> ci_evec;
	  
          Teuchos::RCP<Epetra_MultiVector> mbStateDensities = 
	    Teuchos::rcp( new Epetra_MultiVector(eigenDataToPass->eigenvectorRe->Map(), nCIevals, true )); //zero out
          eigenDataToPass->eigenvalueRe->resize(nCIevals);
          eigenDataToPass->eigenvalueIm->resize(nCIevals);
	  
          //int rank = tcomm->getRank();
          //std::cout << "DEBUG Rank " << rank << ": " << nCIevals << " evals" << std::endl;
	  
          for(int k=0; k < nCIevals; k++) {
	    soln->getEigenvectorPxMatrix(k, mxPx); // mxPx = n1P x n1P matrix of coeffs of 1P products
	    (*(eigenDataToPass->eigenvalueRe))[k] = eigenvalues[k];
	    (*(eigenDataToPass->eigenvalueIm))[k] = 0.0; //evals are real
	    
	    //Note that CI's n1P is twice the number of eigenvalues in Albany eigendata due to spin degeneracy
	    // and we must sum up and down parts [2*i and 2*i+1 ==> spatial evec i] -- LATER: get this info from soln?
	    for(int i=0; i < n1PperBlock; i++) {
	      const Epetra_Vector& vi_real = *((*(eigenDataToPass->eigenvectorRe))(i));
	      const Epetra_Vector& vi_imag = *((*(eigenDataToPass->eigenvectorIm))(i));
	      
	      for(int j=0; j < n1PperBlock; j++) {
		const Epetra_Vector& vj_real = *((*(eigenDataToPass->eigenvectorRe))(j));
		const Epetra_Vector& vj_imag = *((*(eigenDataToPass->eigenvectorIm))(j));
		
		(*mbStateDensities)(k)->Multiply( mxPx[i][j], vi_real, vj_real, 1.0); // mbDen(k) += mxPx_ij * elwise(Vi_r * Vj_r)
		(*mbStateDensities)(k)->Multiply( mxPx[i][j], vi_imag, vj_imag, 1.0); // mbDen(k) += mxPx_ij * elwise(Vi_i * Vj_i)
	      }
	    }
          }
    
          // Put densities into eigenDataToPass (evals already done above):
          //   (just for good measure duplicate in re and im multivecs so they're the same size - probably unecessary)
          eigenDataToPass->eigenvectorRe = mbStateDensities;
          eigenDataToPass->eigenvectorIm = mbStateDensities; 
      }
      else { 
	//if we don't run the CI because there are no particles just 
	// zero out what would be the many body electron densities
	if(bVerbose) *out << "QCAD Solve: Skipping CI solve (no particles)" << std::endl;
	int nCIevals = 0;
	eigenDataToPass->eigenvalueRe->resize(nCIevals);
	eigenDataToPass->eigenvalueIm->resize(nCIevals);
      }
	

      // Poisson Solve which uses CI MB state density and eigenvalues to get quantum electron density
      if(bVerbose) *out << "QCAD Solve: CI Poisson iteration " << iter << std::endl;
      QCAD::SolveModel(getSubSolver("CIPoisson"), pStatesToPass, pStatesToLoop,
		 eigenDataToPass, eigenDataNull);
      
    }
    else {
      
      // Poisson Solve which uses Schrodinger evals & evecs to get quantum electron density
      if(bVerbose) *out << "QCAD Solve: Poisson iteration " << iter << std::endl;
      QCAD::SolveModel(getSubSolver("Poisson"), pStatesToPass, pStatesToLoop,
		 eigenDataToPass, eigenDataNull);


      //assume that a Field Value and then a Field Integral response that computes the total number of electrons in 
      // the quantum regions are the last "component" response functions comprising the aggregated response function
      // that fills response vector 0.  A Field Value response computes 5 doubles, and a Field Integral response
      // computes 1, so the value of the field integral is the 6th element from the end.
      Teuchos::RCP<Epetra_Vector> g = getSubSolver("Poisson").responses_out->get_g(0); //Get poisson solver responses
      int totalQuantumElectronsResponseIndex = g->GlobalLength() - 6;

      if(bVerbose) *out << "QCAD Solve: Poisson iteration has " << (*g)[totalQuantumElectronsResponseIndex] 
			<< " electrons in the quantum region" << std::endl;

      /**out << "DEBUG: Poisson response dump:" << std::endl;
      *out << "g[0] = " << (*g)[0] << std::endl;
      *out << "g[1] = " << (*g)[1] << std::endl;
      *out << "g[2] = " << (*g)[2] << std::endl;
      *out << "g[3] = " << (*g)[3] << std::endl;
      *out << "g[4] = " << (*g)[4] << std::endl;
      *out << "g[5] = " << (*g)[5] << std::endl;
      *out << "g[6] = " << (*g)[6] << std::endl;*/
    }

    eigenDataNull = Teuchos::null;

    if(iter > 1) {
      if(bPoissonSchrodingerConverged == false) {
	double local_maxDiff = QCAD::getMaxDifference(*pStatesToLoop, prevElectricPotential, "PS Saved Electric Potential");
	double global_maxDiff;
	solverComm->MaxAll(&local_maxDiff, &global_maxDiff, 1);
	bPoissonSchrodingerConverged = (global_maxDiff < CONVERGE_TOL);
	
	if(bVerbose) *out << "QCAD Solve: Electric Potential max diff=" 
			  << global_maxDiff << " (tol=" << CONVERGE_TOL << ")" << std::endl;

	
	if(bPoissonSchrodingerConverged) {
	  //Get the number of particles converged upon by the Poisson-Schrodinger loop to use at the number of particles for the CI
	  Teuchos::RCP<Epetra_Vector> g = getSubSolver("Poisson").responses_out->get_g(0); //Get poisson solver responses

	  double nParticlesInQR;
	  int nParticles, nExcitations;
	  int totalQuantumElectronsResponseIndex = g->GlobalLength() - 6; // 6th from end, assuming response ordering (see above)
  	  //nParticles = 2;  //hardcoded for testing

	  nParticlesInQR = (*g)[totalQuantumElectronsResponseIndex];
	  nParticles = std::min((int)round(nParticlesInQR), maxCIParticles); 

	  nExcitations = std::min(nParticles,4); //four excitations at most?
	  MyPL->set("Num Excitations", nExcitations);
	  MyPL->set("Subbasis Particles 0", nParticles);	  

	  if(nParticles <= 0) {
	    bRunCI = false;
	    if(bVerbose) *out << "QCAD Solve: SP Converged.  " << nParticlesInQR << " electrons in QR. "
			      << "Not starting CI since there are no particles (electrons)." << std::endl;
	  }
	  else if(bVerbose) *out << "QCAD Solve: SP Converged.  " << nParticlesInQR << " electrons in QR. Starting CI with " 
			    << nParticles << " particles, " << nExcitations << " excitations" << std::endl;
	}

      }
      else {
	//TODO - convergence criterion for Poisson-CI Loop -- now don't loop at all, just converge right away
	bConverged = true;
      }
    }
      
    QCAD::CopyStateToContainer(*pStatesToLoop, "PS Saved Electric Potential", prevElectricPotential);
  }

  if(bVerbose) {
    if(bConverged)
      *out << "QCAD Solve: Converged Poisson-CI solve loop after " << iter << " iterations." << std::endl;
    else
      *out << "QCAD Solve: Maximum iterations (" << maxIter << ") reached." << std::endl;
  }

  if(bConverged) {
    // LATER: perhaps run a separate Poisson CI solve (as above) but have it compute all the responses we want
    //  (and don't have it compute them in the in-loop call above).

    //TODO: why is this here?
    //Write parameters and responses of final PoissonCI solve
    // Don't worry about sensitivities yet - just output vectors

    const QCAD::SolverSubSolver& ss = getSubSolver("CIPoisson");
    int cipoisson_num_p = ss.params_in->Np();     // Number of *vectors* of parameters
    int cipoisson_num_g = ss.responses_out->Ng(); // Number of *vectors* of responses

    for (int i=0; i<cipoisson_num_p; i++)
      ss.params_in->get_p(i)->Print(*out << "\nCI Parameter vector " << i << ":\n");

    for (int i=0; i<cipoisson_num_g-1; i++) {
      Teuchos::RCP<Epetra_Vector> g = ss.responses_out->get_g(i);
      bool is_scalar = true;

      if (ss.app != Teuchos::null)
        is_scalar = ss.app->getResponse(i)->isScalarResponse();

      if (is_scalar) {
        g->Print(*out << "\nCIPoisson Response vector " << i << ":\n");
	*out << "\n";  //add blank line after vector is printed - needed for proper post-processing
	// see Main_Solve.cpp for how to print sensitivities here
      }
    }

    // Invoke final observer to output evecs (MB densities) and Poisson solution
    Teuchos::RCP<Epetra_Vector> solnVec = getSubSolver("CIPoisson").responses_out->get_g(1); //get the *first* response vector (solution)
    final_obs->observeSolution(*solnVec, "Potential", eigenDataToPass, 0.0);
  }



  #else
  
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       "Albany must be built with ALBANY_CI enabled in order to perform Poisson-CI iterative solutions." << std::endl);

  #endif
}



void QCAD::Solver::setupParameterMapping(const Teuchos::ParameterList& list, const std::string& defaultSubSolver)
{
  std::string s;
  std::vector<std::string> fnStrings;

  if( list.isType<int>("Number") ) {
    nParameters = list.get<int>("Number");

    for(std::size_t i=0; i<nParameters; i++) {
      s = list.get<std::string>(Albany::strint("Parameter",i));
      s = QCAD::string_remove_whitespace(s);
      fnStrings = QCAD::string_split(s,';',true);
    
      std::vector<Teuchos::RCP<QCAD::SolverParamFn> > fnVec;
      for(std::size_t j=0; j<fnStrings.size(); j++)
	fnVec.push_back( Teuchos::rcp(new QCAD::SolverParamFn(fnStrings[j], subSolvers)) );

      paramFnVecs.push_back( fnVec );
    }
  }

  else {  // When "Number" is not given, expose all the parameters of the default SubSolver
    
    const Epetra_Vector& default_solver_p = *((subSolvers.find(defaultSubSolver)->second).params_in->get_p(0));
    nParameters = default_solver_p.MyLength();

    for(std::size_t i=0; i<nParameters; i++) {
      std::ostringstream ss;
      std::vector<Teuchos::RCP<QCAD::SolverParamFn> > fnVec;

      ss << defaultSubSolver << "[" << i << "]";  // "defaultSubSolver[i]"
      fnVec.push_back( Teuchos::rcp(new QCAD::SolverParamFn( ss.str(), subSolvers)) );
      paramFnVecs.push_back( fnVec );
    }
  }
}


void QCAD::Solver::setupResponseMapping(const Teuchos::ParameterList& list, const std::string& defaultSubSolver, int nEigenvalues)
{
  Teuchos::RCP<QCAD::SolverResponseFn> fn;

  if( list.isType<int>("Number") ) {
    int number = list.get<int>("Number");
    std::string s;

    nResponseDoubles = 0;
    for(int i=0; i<number; i++) {
      s = list.get<std::string>(Albany::strint("Response",i));
      fn = Teuchos::rcp(new QCAD::SolverResponseFn(s, subSolvers, nEigenvalues));
      nResponseDoubles += fn->getNumDoubles();
      responseFns.push_back( fn );
    }
  }

  else {   // When "Number" is not given, echo all of the responses of the default SubSolver
    std::ostringstream ss;
    ss << defaultSubSolver << "[:]"; // all responses of defaultSubSolver
    fn = Teuchos::rcp(new QCAD::SolverResponseFn(ss.str(), subSolvers, nEigenvalues));
    nResponseDoubles = fn->getNumDoubles();
    responseFns.push_back( fn );
  }
}

const QCAD::SolverSubSolver& QCAD::Solver::getSubSolver(const std::string& name) const
{
  return subSolvers.find(name)->second;
}

void QCAD::Solver::
preprocessParams(Teuchos::ParameterList& params, std::string preprocessType) const
{
  Teuchos::ParameterList emptyParamlist("Empty Parameters");

  if(preprocessType == "initial poisson") {
    //! Turn off sensitivities
    setRequestSensitivities(params, false);

    //! Set poisson parameters
    params.sublist("Problem").sublist("Poisson Source").set("Quantum Region Source", "semiclassical");
    params.sublist("Problem").sublist("Poisson Source").set("Non Quantum Region Source", "semiclassical");

    //! Rename output file
    if (params.sublist("Discretization").isParameter("Exodus Output File Name"))
    {
      std::string exoName= params.sublist("Discretization").get<std::string>("Exodus Output File Name") + ".init";
      params.sublist("Discretization").set("Exodus Output File Name", exoName);
    }
    else if (params.sublist("Discretization").isParameter("1D Output File Name"))
    {
      std::string exoName= "init" + params.sublist("Discretization").get<std::string>("1D Output File Name");
      params.sublist("Discretization").set("1D Output File Name", exoName);
    }
    
    else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			  "Unknown function Discretization Parameter" << std::endl);
  }

  else if(preprocessType == "CI poisson") {
    //! Rename output file -- NO LONGER: keep "root" exo name for CI poisson solver, as it will be final result
    //std::string exoName= "ci" + params.sublist("Discretization").get<std::string>("Exodus Output File Name");
    //std::string exoName= params.sublist("Discretization").get<std::string>("Exodus Output File Name") + ".ci";
    //params.sublist("Discretization").set("Exodus Output File Name", exoName);

    //! Set poisson parameters
    params.sublist("Problem").sublist("Poisson Source").set("Quantum Region Source", "ci");
    params.sublist("Problem").sublist("Poisson Source").set("Non Quantum Region Source", "semiclassical");
  }

  else if(preprocessType == "Delta poisson") {
    //! Turn off sensitivities
    setRequestSensitivities(params, false);

    //! Rename output file
    //std::string exoName= "delta" + params.sublist("Discretization").get<std::string>("Exodus Output File Name");
    std::string exoName= params.sublist("Discretization").get<std::string>("Exodus Output File Name") + ".delta";
    params.sublist("Discretization").set("Exodus Output File Name", exoName);

    //! Set poisson parameters
    params.sublist("Problem").sublist("Poisson Source").set("Quantum Region Source", "schrodinger");
    params.sublist("Problem").sublist("Poisson Source").set("Non Quantum Region Source", "none");

    //! Set responses: add responses for each pair ( evec_i, evec_j ) => delta_ij
    Teuchos::ParameterList& responseList = params.sublist("Problem").sublist("Response Functions");
    int nEigenvectors = params.sublist("Problem").sublist("Poisson Source").get<int>("Eigenvectors from States"); 
    int initial_nResponses = responseList.get<int>("Number"); //Shift existing responses
    int added_nResponses = 2 * nEigenvectors * (nEigenvectors + 1) / 2;  //mult by 2 for real & imag parts
    char buf1[200], buf2[200], buf1i[200], buf2i[200];
    int iResponse;
    responseList.set("Number", initial_nResponses + added_nResponses); //sets member

    //shift response indices of existing responses by added_responses so added responses index from zero
    for(int i=initial_nResponses-1; i >= 0; i--) {       
      std::string respType = responseList.get<std::string>(Albany::strint("Response",i));
      responseList.set(Albany::strint("Response",i + added_nResponses), respType);
      responseList.sublist( Albany::strint("ResponseParams",i + added_nResponses) ) = 
	Teuchos::ParameterList(responseList.sublist( Albany::strint("ResponseParams",i) ) ); //create new copy of list
      responseList.sublist( Albany::strint("ResponseParams",i) ) = Teuchos::ParameterList(Albany::strint("ResponseParams",i)); //clear sublist i
    }

    iResponse = 0;
    for(int i=0; i<nEigenvectors; i++) {
      sprintf(buf1, "%s_Re%d", "Evec", i);
      sprintf(buf1i, "%s_Im%d", "Evec", i);
      for(int j=i; j<nEigenvectors; j++) {
	sprintf(buf2, "%s_Re%d", "Evec", j);
	sprintf(buf2i, "%s_Im%d", "Evec", j);

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams.set("Field Name", "Electric Potential");  // same as solution, but at quad points and referenced to 0
	responseParams.set("Field Name 1", buf1);  
	responseParams.set("Field Name Im 1", buf1i);
	responseParams.set("Field Name 2", buf2);
	responseParams.set("Field Name Im 2", buf2i);
	responseParams.set("Conjugate Field 1", true);
	responseParams.set("Conjugate Field 2", false);
	responseParams.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams.set("Return Imaginary Part", false);

	iResponse++;

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams2 = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams2.set("Field Name", "Electric Potential");  // same as solution, but at quad points and referenced to 0
	responseParams2.set("Field Name 1", buf1);
	responseParams2.set("Field Name Im 1", buf1i);
	responseParams2.set("Field Name 2", buf2);
	responseParams2.set("Field Name Im 2", buf2i);
	responseParams2.set("Conjugate Field 1", true);
	responseParams2.set("Conjugate Field 2", false);
	responseParams2.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams2.set("Return Imaginary Part", true);

	iResponse++;
      }
    }
  }

  else if(preprocessType == "No charge poisson") {
    //! Turn off sensitivities
    setRequestSensitivities(params, false);

    //! Rename output file
    std::string exoName= params.sublist("Discretization").get<std::string>("Exodus Output File Name") + ".nocharge";
    params.sublist("Discretization").set("Exodus Output File Name", exoName);

    //! Set poisson parameters
    params.sublist("Problem").sublist("Poisson Source").set("Quantum Region Source", "none");
    params.sublist("Problem").sublist("Poisson Source").set("Non Quantum Region Source", "none");

    //! Set responses: add responses for each pair ( evec_i, evec_j ) => Coulomb_noSrcCharge(i,j)
    Teuchos::ParameterList& responseList = params.sublist("Problem").sublist("Response Functions");
    int nEigenvectors = params.sublist("Problem").sublist("Poisson Source").get<int>("Eigenvectors from States"); 
    int initial_nResponses = responseList.get<int>("Number");
    int added_nResponses = 2 * nEigenvectors * (nEigenvectors + 1) / 2;  //mult by 2 for real & imag parts
    char buf1[200], buf2[200], buf1i[200], buf2i[200];
    int iResponse;

    responseList.set("Number", initial_nResponses + nEigenvectors * (nEigenvectors + 1) / 2);

    //shift response indices of existing responses by added_responses so added responses index from zero
    for(int i=initial_nResponses-1; i >= 0; i--) {       
      std::string respType = responseList.get<std::string>(Albany::strint("Response",i));
      responseList.set(Albany::strint("Response",i + added_nResponses), respType);
      responseList.sublist( Albany::strint("ResponseParams",i + added_nResponses) ) = 
	Teuchos::ParameterList(responseList.sublist( Albany::strint("ResponseParams",i)) ); 
      responseList.sublist( Albany::strint("ResponseParams",i) ) = Teuchos::ParameterList(Albany::strint("ResponseParams",i)); //clear sublist i
    }

    iResponse = 0;
    for(int i=0; i<nEigenvectors; i++) {
      sprintf(buf1, "%s_Re%d", "Evec", i);
      sprintf(buf1i, "%s_Im%d", "Evec", i);
      for(int j=i; j<nEigenvectors; j++) {
	sprintf(buf2, "%s_Re%d", "Evec", j);
	sprintf(buf2i, "%s_Im%d", "Evec", j);

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams.set("Field Name", "Electric Potential");  // same as solution, but at quad points and referenced to 0
	responseParams.set("Field Name 1", buf1);
	responseParams.set("Field Name Im 1", buf1i);
	responseParams.set("Field Name 2", buf2);
	responseParams.set("Field Name Im 2", buf2i);
	responseParams.set("Conjugate Field 1", true);
	responseParams.set("Conjugate Field 2", false);
	responseParams.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams.set("Return Imaginary Part", false);
	
	iResponse++;

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams2 = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams2.set("Field Name", "Electric Potential");  // same as solution, but at quad points and referenced to 0
	responseParams2.set("Field Name 1", buf1);
	responseParams2.set("Field Name Im 1", buf1i);
	responseParams2.set("Field Name 2", buf2);
	responseParams2.set("Field Name Im 2", buf2i);
	responseParams2.set("Conjugate Field 1", true);
	responseParams2.set("Conjugate Field 2", false);
	responseParams2.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams2.set("Return Imaginary Part", true);
	
	iResponse++;
      }
    }
  }
	
  else if(preprocessType == "Coulomb poisson") {
    //! Turn off sensitivities
    setRequestSensitivities(params, false);

    //! Rename output file
    //std::string exoName= "coulomb" + params.sublist("Discretization").get<std::string>("Exodus Output File Name");
    std::string exoName= params.sublist("Discretization").get<std::string>("Exodus Output File Name") + ".coulomb";
    params.sublist("Discretization").set("Exodus Output File Name", exoName);

    //! Set poisson parameters
    params.sublist("Problem").sublist("Poisson Source").set("Quantum Region Source", "coulomb");
    params.sublist("Problem").sublist("Poisson Source").set("Non Quantum Region Source", "none");
    params.sublist("Problem").sublist("Poisson Source").set("Imaginary Part Of Coulomb Source", false);

    //! Specify source eigenvector indices as parameters
    int nParams = params.sublist("Problem").sublist("Parameters").get<int>("Number");
    params.sublist("Problem").sublist("Parameters").set("Number", nParams + 2); //assumes Source Eigenvector X are not already params
    params.sublist("Problem").sublist("Parameters").set(Albany::strint("Parameter",nParams), "Source Eigenvector 1");
    params.sublist("Problem").sublist("Parameters").set(Albany::strint("Parameter",nParams+1), "Source Eigenvector 2");

    //! Set responses: add responses for each pair ( evec_i, evec_j ) => Coulomb(i,src_evec1,j,src_evec2)
    Teuchos::ParameterList& responseList = params.sublist("Problem").sublist("Response Functions");
    int nEigenvectors = params.sublist("Problem").sublist("Poisson Source").get<int>("Eigenvectors from States");
    int initial_nResponses = responseList.get<int>("Number");
    int added_nResponses = 2 * nEigenvectors * (nEigenvectors + 1) / 2;  //mult by 2 for real & imag parts
    char buf1[200], buf2[200], buf1i[200], buf2i[200];
    int iResponse;

    responseList.set("Number", initial_nResponses + added_nResponses);

    //shift response indices of existing responses by added_responses so added responses index from zero
    for(int i=initial_nResponses-1; i >= 0; i--) {       
      std::string respType = responseList.get<std::string>(Albany::strint("Response",i));
      responseList.set(Albany::strint("Response",i + added_nResponses), respType);
      responseList.sublist( Albany::strint("ResponseParams",i + added_nResponses) ) = 
	Teuchos::ParameterList(responseList.sublist( Albany::strint("ResponseParams",i)) ); 
      responseList.sublist( Albany::strint("ResponseParams",i) ) = Teuchos::ParameterList(Albany::strint("ResponseParams",i)); //clear sublist i
    }

    iResponse = 0;
    for(int i=0; i<nEigenvectors; i++) {
      sprintf(buf1, "%s_Re%d", "Evec", i);
      sprintf(buf1i, "%s_Im%d", "Evec", i);
      for(int j=i; j<nEigenvectors; j++) {
	sprintf(buf2, "%s_Re%d", "Evec", j);
	sprintf(buf2i, "%s_Im%d", "Evec", j);

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams.set("Field Name", "Electric Potential");  // same as solution, but at quad points and referenced to 0
	responseParams.set("Field Name 1", buf1);
	responseParams.set("Field Name Im 1", buf1i);
	responseParams.set("Field Name 2", buf2);
	responseParams.set("Field Name Im 2", buf2i);
	responseParams.set("Conjugate Field 1", true);
	responseParams.set("Conjugate Field 2", false);
	responseParams.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams.set("Return Imaginary Part", false);
	
	iResponse++;

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams2 = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams2.set("Field Name", "Electric Potential");  // same as solution, but at quad points and referenced to 0
	responseParams2.set("Field Name 1", buf1);
	responseParams2.set("Field Name Im 1", buf1i);
	responseParams2.set("Field Name 2", buf2);
	responseParams2.set("Field Name Im 2", buf2i);
	responseParams2.set("Conjugate Field 1", true);
	responseParams2.set("Conjugate Field 2", false);
	responseParams2.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams2.set("Return Imaginary Part", true);
	
	iResponse++;
      }
    }
  }

  else if(preprocessType == "Coulomb poisson imaginary") {
    //! Turn off sensitivities
    setRequestSensitivities(params, false);

    //! Rename output file
    //std::string exoName= "coulomb" + params.sublist("Discretization").get<std::string>("Exodus Output File Name");
    std::string exoName= params.sublist("Discretization").get<std::string>("Exodus Output File Name") + ".coulomb";
    params.sublist("Discretization").set("Exodus Output File Name", exoName);

    //! Set poisson parameters
    params.sublist("Problem").sublist("Poisson Source").set("Quantum Region Source", "coulomb");
    params.sublist("Problem").sublist("Poisson Source").set("Non Quantum Region Source", "none");
    params.sublist("Problem").sublist("Poisson Source").set("Imaginary Part Of Coulomb Source", true);

    //! Specify source eigenvector indices as parameters
    int nParams = params.sublist("Problem").sublist("Parameters").get<int>("Number");
    params.sublist("Problem").sublist("Parameters").set("Number", nParams + 2); //assumes Source Eigenvector X are not already params
    params.sublist("Problem").sublist("Parameters").set(Albany::strint("Parameter",nParams), "Source Eigenvector 1");
    params.sublist("Problem").sublist("Parameters").set(Albany::strint("Parameter",nParams+1), "Source Eigenvector 2");

    //! Set responses: add responses for each pair ( evec_i, evec_j ) => Coulomb(i,src_evec1,j,src_evec2)
    Teuchos::ParameterList& responseList = params.sublist("Problem").sublist("Response Functions");
    int nEigenvectors = params.sublist("Problem").sublist("Poisson Source").get<int>("Eigenvectors from States");
    int initial_nResponses = responseList.get<int>("Number");
    int added_nResponses = 2 * nEigenvectors * (nEigenvectors + 1) / 2;  //mult by 2 for real & imag parts
    char buf1[200], buf2[200], buf1i[200], buf2i[200];
    int iResponse;

    responseList.set("Number", initial_nResponses + added_nResponses);

    //shift response indices of existing responses by added_responses so added responses index from zero
    for(int i=initial_nResponses-1; i >= 0; i--) {       
      std::string respType = responseList.get<std::string>(Albany::strint("Response",i));
      responseList.set(Albany::strint("Response",i + added_nResponses), respType);
      responseList.sublist( Albany::strint("ResponseParams",i + added_nResponses) ) = 
	Teuchos::ParameterList(responseList.sublist( Albany::strint("ResponseParams",i)) ); 
      responseList.sublist( Albany::strint("ResponseParams",i) ) = Teuchos::ParameterList(Albany::strint("ResponseParams",i)); //clear sublist i
    }

    iResponse = 0;
    for(int i=0; i<nEigenvectors; i++) {
      sprintf(buf1, "%s_Re%d", "Evec", i);
      sprintf(buf1i, "%s_Im%d", "Evec", i);
      for(int j=i; j<nEigenvectors; j++) {
	sprintf(buf2, "%s_Re%d", "Evec", j);
	sprintf(buf2i, "%s_Im%d", "Evec", j);

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams.set("Field Name", "Electric Potential");  // same as solution, but must be at quad points
	responseParams.set("Field Name 1", buf1);
	responseParams.set("Field Name Im 1", buf1i);
	responseParams.set("Field Name 2", buf2);
	responseParams.set("Field Name Im 2", buf2i);
	responseParams.set("Conjugate Field 1", true);
	responseParams.set("Conjugate Field 2", false);
	responseParams.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams.set("Return Imaginary Part", false);
	
	iResponse++;

	responseList.set(Albany::strint("Response",iResponse), "Field Integral");
	Teuchos::ParameterList& responseParams2 = responseList.sublist(Albany::strint("ResponseParams",iResponse));
	responseParams2.set("Field Name", "Electric Potential");  // same as solution, but must be at quad points
	responseParams2.set("Field Name 1", buf1);
	responseParams2.set("Field Name Im 1", buf1i);
	responseParams2.set("Field Name 2", buf2);
	responseParams2.set("Field Name Im 2", buf2i);
	responseParams2.set("Conjugate Field 1", true);
	responseParams2.set("Conjugate Field 2", false);
	responseParams2.set("Integrand Length Unit", "mesh"); // same as mesh
	responseParams2.set("Return Imaginary Part", true);
	
	iResponse++;
      }
    }
  }
  
  else if(preprocessType == "Poisson") {
    //! Turn off sensitivities
    setRequestSensitivities(params, false);

    //! Rename output file
    std::string exoName= params.sublist("Discretization").get<std::string>("Exodus Output File Name") + ".poisson";
    params.sublist("Discretization").set("Exodus Output File Name", exoName);
  }


}

int QCAD::Solver::ExtractNumberOfEigenvectors(const std::string xmlfilename, const Epetra_Comm& comm) const
{
  const Albany_MPI_Comm mpiComm = Albany::getMpiCommFromEpetraComm(comm);
  Teuchos::RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(mpiComm);
  Teuchos::RCP<Teuchos::ParameterList> appParams = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromXmlFileAndBroadcast(xmlfilename, appParams.ptr(), *tcomm);
  return appParams->sublist("Problem").sublist("Poisson Source").get<int>("Eigenvectors from States"); 
}


QCAD::SolverSubSolver 
QCAD::Solver::CreateSubSolver(const Teuchos::RCP<Teuchos::ParameterList> appParams, const Epetra_Comm& comm,
			      const Teuchos::RCP<const Epetra_Vector>& initial_guess) const
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  QCAD::SolverSubSolver ret; //value to return

  const Albany_MPI_Comm mpiComm = Albany::getMpiCommFromEpetraComm(comm);

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "QCAD Solver creating solver from " << appParams->name() 
       << " parameter list" << std::endl;
 
  //! Create solver factory, which reads xml input filen
  Albany::SolverFactory slvrfctry(appParams, mpiComm);
    
  //! Create solver and application objects via solver factory
  RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(mpiComm);
  ret.model = slvrfctry.createAndGetAlbanyApp(ret.app, appComm, appComm, initial_guess);

  ret.params_in = rcp(new EpetraExt::ModelEvaluator::InArgs);
  ret.responses_out = rcp(new EpetraExt::ModelEvaluator::OutArgs);  

  *(ret.params_in) = ret.model->createInArgs();
  *(ret.responses_out) = ret.model->createOutArgs();
  int ss_num_p = ret.params_in->Np();     // Number of *vectors* of parameters
  int ss_num_g = ret.responses_out->Ng(); // Number of *vectors* of responses
  RCP<Epetra_Vector> p1;
  RCP<Epetra_Vector> g1;
  
  if (ss_num_p > 0)
    p1 = rcp(new Epetra_Vector(*(ret.model->get_p_init(0))));
  if (ss_num_g > 1)
    g1 = rcp(new Epetra_Vector(*(ret.model->get_g_map(0))));
  RCP<Epetra_Vector> xfinal =
    rcp(new Epetra_Vector(*(ret.model->get_g_map(ss_num_g-1)),true) );
  
  // Sensitivity Analysis stuff
  bool supportsSensitivities = false;
  RCP<Epetra_MultiVector> dgdp;
  
  if (ss_num_p>0 && ss_num_g>1) {
    supportsSensitivities =
      !ret.responses_out->supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp, 0, 0).none();
    
    if (supportsSensitivities) {
      if (p1->GlobalLength() > 0)
        dgdp = rcp(new Epetra_MultiVector(g1->Map(), p1->GlobalLength() ));
      else
        supportsSensitivities = false;
    }
  }
  
  if (ss_num_p > 0)  ret.params_in->set_p(0,p1);
  if (ss_num_g > 1)  ret.responses_out->set_g(0,g1);
  ret.responses_out->set_g(ss_num_g-1,xfinal);
  
  if (supportsSensitivities) ret.responses_out->set_DgDp(0,0,dgdp);
  
  return ret;
}

void QCAD::Solver::SetCoulombParams(const Teuchos::RCP<EpetraExt::ModelEvaluator::InArgs> inArgs, int i2, int i4) const
{
  TEUCHOS_TEST_FOR_EXCEPTION( inArgs->Np() < 1, Teuchos::Exceptions::InvalidParameter, 
			      "Cannot set coulomb parameters because there are no parameter vectors.");
  Teuchos::RCP<const Epetra_Vector> p_ro = inArgs->get_p(0); //only use *first* param vector now
  Teuchos::RCP<Epetra_Vector> p = Teuchos::rcp( new Epetra_Vector( *p_ro ) );
  
  // assume the last two parameters are i2 and i4 -- indices for the coulomb element to be computed
  std::size_t nParams = p->GlobalLength();
  (*p)[ nParams-2 ] = (double) i2;
  (*p)[ nParams-1 ] = (double) i4;

  inArgs->set_p(0, p);
}

void QCAD::Solver::setRequestSensitivities(Teuchos::ParameterList &params, bool flag)
{
  Teuchos::ParameterList &solveParams = params.sublist("Piro").sublist("Analysis").sublist("Solve");
  solveParams.set("Compute Sensitivities", flag);
}



Teuchos::RCP<const Teuchos::ParameterList>
QCAD::Solver::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::createParameterList("ValidPoissonSchrodingerProblemParams");

  validPL->set<std::string>("Name", "", "String to designate Problem");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0,
                    "Flag to select output of Phalanx Graph and level of detail");
  validPL->set<bool>("Verbose Output",false,"Enable detailed output mode");

  validPL->set<double>("LengthUnitInMeters",1e-6,"Length unit in meters");
  validPL->set<double>("EnergyUnitInElectronVolts",1.0,"Energy (voltage) unit in electron volts (volts)");
  validPL->set<double>("Temperature",300,"Temperature in Kelvin");
  validPL->set<std::string>("MaterialDB Filename","materials.xml","Filename of material database xml file");

  validPL->set<bool>("Use Integrated Poisson Schrodinger",true,"After converging iterative P-S, run integrated P-S solver");
  validPL->set<int>("Number of Eigenvalues",0,"The number of eigenvalue-eigenvector pairs");
  validPL->set<int>("Maximum Iterations",100,"The maximum number of P-S iterations");
  validPL->set<double>("Eigensolver Percent Shift Below Potential Min", 1.0, "Percentage of energy range of potential to subtract from the potential's minimum to obtain the eigensolver's shift");
  validPL->set<double>("Iterative PS Convergence Tolerance", 1e-6, "Convergence criterion for iterative PS solver (max potential difference across mesh)");

  validPL->set<int>("Maximum CI Particles", 0, "Poisson Schrodinger CI mode only: the maximum number of particles to use in the CI phase");
  validPL->set<int>("CI Particles", 0, "Schrodinger CI mode only: the number of particles to use in the CI phase");
  validPL->set<int>("CI Excitations", 0, "Schrodinger CI mode only: the number of excitations with which to truncate the CI phase");

  validPL->set<bool>("Include exchange-correlation potential",false,"Include exchange-correlation potential in poisson source term");
  validPL->set<bool>("Only solve schrodinger in quantum blocks",true,"Limit schrodinger solution to elements blocks labeled as quantum in the materials DB");

  validPL->set<bool>("Use predictor-corrector method",false,"Enable the predictor corrector algorithm for P-S iteration (otherwise use Picard)");

  validPL->sublist("Poisson Problem", false, "");
  validPL->sublist("Schrodinger Problem", false, "");

  // Validate Parameters
  const int maxParameters = 100;
  Teuchos::ParameterList& validParamPL = validPL->sublist("Parameters", false, "");
  validParamPL.set<int>("Number", 0);
  for (int i=0; i<maxParameters; i++) {
    validParamPL.set<std::string>(Albany::strint("Parameter",i), "");
  }

  // Validate Responses
  const int maxResponses = 100;
  Teuchos::ParameterList& validResponsePL = validPL->sublist("Response Functions", false, "");
  validResponsePL.set<int>("Number of Response Vectors", 0);
  validResponsePL.set<int>("Number", 0);
  validResponsePL.set<int>("Equation", 0);
  for (int i=0; i<maxResponses; i++) {
    validResponsePL.set<std::string>(Albany::strint("Response",i), "");
    validResponsePL.sublist(Albany::strint("ResponseParams",i));
    validResponsePL.sublist(Albany::strint("Response Vector",i));
  }

  // Candidates for deprecation. Pertain to the solution rather than the problem definition.
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");
  
  return validPL;
}



// Parameter Function object

// Function string can be of form:
// "fn1(a,b)>fn2(c,d) ... >SolverName[X:Y]  OR
// "SolverName[X:Y]"
QCAD::SolverParamFn::SolverParamFn(const std::string& fnString, 
			     const std::map<std::string, QCAD::SolverSubSolver>& subSolvers)
{
  std::vector<std::string> fnsAndTarget = QCAD::string_split(fnString,'>',true);
  std::vector<std::string>::const_iterator it;  
  std::map<std::string,std::string> target;

  if( fnsAndTarget.begin() != fnsAndTarget.end() ) {
    for(it=fnsAndTarget.begin(); it != fnsAndTarget.end()-1; it++) {
      filters.push_back( QCAD::string_parse_function( *it ) );
    }
    it = fnsAndTarget.end()-1;
    target = QCAD::string_parse_arrayref( *it );
    targetName = target["name"];
    
    const Epetra_Vector& solver_p = *((subSolvers.find(targetName)->second).params_in->get_p(0));
    targetIndices = QCAD::string_expand_compoundindex(target["index"], 0, solver_p.MyLength());
  } 
}


void QCAD::SolverParamFn::fillSubSolverParams(double parameterValue, 
   const std::map<std::string, QCAD::SolverSubSolver>& subSolvers) const
{
  std::vector< std::vector<std::string> >::const_iterator fit;
  for(fit = filters.begin(); fit != filters.end(); ++fit) {

    //perform function operation
    std::string fnName = (*fit)[0];
    if( fnName == "scale" ) {
      TEUCHOS_TEST_FOR_EXCEPT( fit->size() != 1+1 ); // "scale" should have 1 parameter
      parameterValue *= atof( (*fit)[1].c_str() );
    }
    else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
	      "Unknown function " << (*fit)[0] << " for given type." << std::endl);
  }

  Teuchos::RCP<EpetraExt::ModelEvaluator::InArgs> inArgs = (subSolvers.find(targetName)->second).params_in;
  Teuchos::RCP<const Epetra_Vector> p_ro = inArgs->get_p(0); //only use *first* param vector now
  Teuchos::RCP<Epetra_Vector> p = Teuchos::rcp( new Epetra_Vector( *p_ro ) );
  
  // copy parameterValue into sub-solver parameter vector where appropriate
  std::vector<int>::const_iterator it;
  for(it = targetIndices.begin(); it != targetIndices.end(); ++it)
    (*p)[ (*it) ] = parameterValue;

  inArgs->set_p(0, p);
}

double QCAD::SolverParamFn::
getInitialParam(const std::map<std::string, QCAD::SolverSubSolver>& subSolvers) const
{
  //get first target parameter's initial value
  double initVal;

  Teuchos::RCP<const Epetra_Vector> p_init = 
    (subSolvers.find(targetName)->second).model->get_p_init(0); //only one p vector used

  TEUCHOS_TEST_FOR_EXCEPT(targetIndices.size() == 0);
  initVal = (*p_init)[ targetIndices[0] ];

  std::vector< std::vector<std::string> >::const_iterator fit;
  for(fit = filters.end(); fit != filters.begin(); --fit) {

    //perform INVERSE function operation to back out initial value
    std::string fnName = (*fit)[0];
    if( fnName == "scale" ) {
      TEUCHOS_TEST_FOR_EXCEPT( fit->size() != 1+1 ); // "scale" should have 1 parameter
      initVal /= atof( (*fit)[1].c_str() );
    }
    else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
	      "Unknown function " << (*fit)[0] << " for given type." << std::endl);
  }

  return initVal;
}

double QCAD::SolverParamFn::
getFilterScaling() const
{
  double scaling = 1.0;

  std::vector< std::vector<std::string> >::const_iterator fit;
  for(fit = filters.end(); fit != filters.begin(); --fit) {

    //perform INVERSE function operation to back out initial value
    std::string fnName = (*fit)[0];
    if( fnName == "scale" ) {
      TEUCHOS_TEST_FOR_EXCEPT( fit->size() != 1+1 ); // "scale" should have 1 parameter
      scaling *= atof( (*fit)[1].c_str() );
    }
  }
  return scaling;
}



// Response Function object

// Function string can be of form:
// "fn1(a,SolverName[X:Y],b)  OR
// "SolverName[X:Y]"
QCAD::SolverResponseFn::SolverResponseFn(const std::string& fnString, 
					 const std::map<std::string, QCAD::SolverSubSolver>& subSolvers,
					 int nEigenvalues)
{
  using std::string;
  std::vector<std::string> fnsAndTarget = QCAD::string_split(fnString,'>',true);
  std::vector<std::string>::const_iterator it;  
  std::map<std::string,std::string> arrayRef;
  ArrayRef ar;
  int nParams = 0;    

  //Case: no function name given
  if( fnString.find_first_of('(') == string::npos ) { 
    fnName = "nop";
    arrayRef = QCAD::string_parse_arrayref( fnString );
    ar.name = arrayRef["name"];

    if(ar.name == "Eigenvalue") {
      ar.indices = QCAD::string_expand_compoundindex(arrayRef["index"], 0, nEigenvalues);
    }
    else {
      const Epetra_Vector& solver_g = *((subSolvers.find(ar.name)->second).responses_out->get_g(0));
      ar.indices = QCAD::string_expand_compoundindex(arrayRef["index"], 0, solver_g.MyLength());
    }
    params.push_back(ar);
    nParams += ar.indices.size();
  }

  //Case: function name given
  else {
    std::vector<std::string> fnAndParams = QCAD::string_parse_function( fnString );
    fnName = fnAndParams[0];

    for(std::size_t i=1; i<fnAndParams.size(); i++) {

      //if contains [ treat as solver array reference, otherwise as a string param
      if( fnAndParams[i].find_first_of('[') == string::npos ) {
	ar.name = fnAndParams[i];  
	ar.indices.clear();
	params.push_back(ar);
	nParams += 1;
      }
      else {
	arrayRef = QCAD::string_parse_arrayref( fnAndParams[i] );
	ar.name = arrayRef["name"];

	const Epetra_Vector& solver_g = *((subSolvers.find(ar.name)->second).responses_out->get_g(0));
	ar.indices = QCAD::string_expand_compoundindex(arrayRef["index"], 0, solver_g.MyLength());

	params.push_back(ar);
	nParams += ar.indices.size();
      }      
    }
  }

  // validate: check number of params and set numDoubles
  if( fnName == "min" || fnName == "max") {
    TEUCHOS_TEST_FOR_EXCEPT(nParams != 2);
    numDoubles = 1;
  }
  else if( fnName == "dist") {
    TEUCHOS_TEST_FOR_EXCEPT( !(nParams == 2 || nParams == 4 || nParams == 6));
    numDoubles = 1;
  }
  else if( fnName == "scale") {
    TEUCHOS_TEST_FOR_EXCEPT(nParams != 2);
    numDoubles = 1;
  }
  else if( fnName == "divide") {
    TEUCHOS_TEST_FOR_EXCEPT(nParams != 2);
    numDoubles = 1;
  }
  else if( fnName == "nop") {
    numDoubles = nParams; 
  }
  else if( fnName == "DgDp") {  //params = subSolverName, pIndex, gIndex
    TEUCHOS_TEST_FOR_EXCEPT(nParams != 3);
    numDoubles = 1; 
  }
  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
     "Unknown function " << fnName << " for QCAD solver response." << std::endl);

}


void QCAD::SolverResponseFn::fillSolverResponses(Epetra_Vector& g, Teuchos::RCP<Epetra_MultiVector>& dgdp, int offset,
				 const std::map<std::string, QCAD::SolverSubSolver>& subSolvers,
				 const std::vector<std::vector<Teuchos::RCP<QCAD::SolverParamFn> > >& paramFnVecs,
				 bool bSupportDpDg, const std::vector<double>& eigenvalueResponses) const
{
  std::size_t nParameters = paramFnVecs.size();

  //Note: for now assume vectors use a local map (later fix using import to local map)
  TEUCHOS_TEST_FOR_EXCEPTION(g.DistributedGlobal(), Teuchos::Exceptions::InvalidParameter,
			       "Error! Solvers's g response vector is distributed.  No implementation for this yet."
			       << std::endl);
  if(dgdp != Teuchos::null)
    TEUCHOS_TEST_FOR_EXCEPTION(dgdp->DistributedGlobal(), Teuchos::Exceptions::InvalidParameter,
			       "Error! Solvers's DgDp multivector is distributed.  No implementation for this yet."
			       << std::endl);

  //Collect values and derivatives (wrt parameters) of function arguments
  std::vector< double > arg_vals; // argument values
  std::vector< std::vector<double> > arg_DgDps; // derivs of arguments (w/possibly mulitple parts) wrt params    

  // Note: arg_DgDps is transposed from Multivector dgdp objects:  
  //   arg_DgDps[ responseIndex ][ paramIndex ]  where dgdp_MultiVector(vectorIndex=paramIndex) [ rowIndex=responseIndex ] 

  std::vector< ArrayRef >::const_iterator arg_it;
  std::string dgdpName;

  //Note "params" == arguments to response function
  for(arg_it = params.begin(); arg_it != params.end(); ++arg_it) {

    if( fnName == "DgDp" && arg_it == params.begin() ) {  //special: string as first parameter to dgdp function
      dgdpName = arg_it->name; continue;
    }

    if( arg_it->indices.size() == 0 ){     //if no indices, "array name" is a double param value

      //single response with double value and zero deriviative  
      arg_vals.push_back( atof(arg_it->name.c_str()) ); 

      if(dgdp != Teuchos::null) {
	std::vector<double> dgdp_accum(nParameters,0.0);
	arg_DgDps.push_back( dgdp_accum );
      }
    }
    else {  // indices => this argument is of form subSolverName[possiblyCompoundIndex]

      std::string solverName = arg_it->name;
      Teuchos::RCP<Epetra_Vector> sub_g;
      Teuchos::RCP<Epetra_MultiVector> sub_dgdp;


      if(solverName == "Eigenvalue") {   //special case of "Eigenvalue[i]"
	std::vector<int>::const_iterator it;
	for(it = arg_it->indices.begin(); it != arg_it->indices.end(); ++it) {
	  arg_vals.push_back( eigenvalueResponses[ *it ]); // append eigenvalue response value
	  if(dgdp != Teuchos::null) {
	    std::vector<double> dgdp_accum(nParameters,0.0); // no sensitivities wrt eigenvalues yet... (all zero)
	    arg_DgDps.push_back( dgdp_accum );
	  }
	}
      }

      else {
	const QCAD::SolverSubSolver& solver = subSolvers.find(solverName)->second;
	sub_g = solver.responses_out->get_g(0); // only use first g vector
	if(dgdp != Teuchos::null)
	  sub_dgdp = solver.responses_out->get_DgDp(0,0).getMultiVector(); // only use first g & p vectors

	// for each index (i.e. double response value) 
	std::vector<int>::const_iterator it; int iIndx;
	for(it = arg_it->indices.begin(), iIndx=0; it != arg_it->indices.end(); ++it, ++iIndx) {
	  int gIndex = *it;
	  arg_vals.push_back( (*sub_g)[ gIndex ]); // append response value
	
	  // append response derivative wrt to each parameter
	  if(dgdp != Teuchos::null) {
	    std::vector<double> dgdp_accum(nParameters,0.0);
	    for(std::size_t i=0; i<nParameters; i++) {
	      const std::vector<Teuchos::RCP<QCAD::SolverParamFn> >& paramFnVec = paramFnVecs[i];
	      for(std::size_t j=0; j<paramFnVec.size(); j++) {
		const QCAD::SolverParamFn& paramFn = *(paramFnVec[j]);
		
		std::string paramTargetName = paramFn.getTargetName();
		const std::vector<int>& paramTargetIndices = paramFn.getTargetIndices();
		double scaling = paramFn.getFilterScaling(); // Later: something more general?
		
		if(paramTargetName != solverName) continue;	
		
		//for each index of the jth element of the ith parameter
		std::vector<int>::const_iterator vit;
		for(vit = paramTargetIndices.begin(); vit != paramTargetIndices.end(); ++vit)
		  dgdp_accum[ i ] += (*((*sub_dgdp)( *vit )))[ gIndex ] * scaling;
	      }
	    }
	    arg_DgDps.push_back( dgdp_accum );
	  }
	}
      }
    }
  } //end of loop over arguments

  // Loop (Pseudocode)
  //for(i=0; i<nParameters; i++) {
  //  dgdp_vecs[iArgument][ i ] = targetSubSolver.DgDp(0,0)[argument_gIndex][ paramIndices[argumentTargetName][i] ];
  //}


  //! Process functions using arg_vals and arg_DpDgs
  std::size_t nArgs = arg_vals.size();

  // minimum
  if( fnName == "min" ) {
    int winIndex = (arg_vals[0] <= arg_vals[1]) ? 0 : 1;      
    g[offset] = arg_vals[winIndex]; //set value

    if(dgdp != Teuchos::null) { //set derivative
      for(std::size_t i=0; i < arg_DgDps[winIndex].size(); i++) 
	dgdp->ReplaceGlobalValue(offset, i, arg_DgDps[winIndex][i]);
    }
  }

  // maximum
  else if(fnName == "max") {
    int winIndex = (arg_vals[0] >= arg_vals[1]) ? 0 : 1;      
    g[offset] = arg_vals[winIndex]; //set value

    if(dgdp != Teuchos::null) { //set derivative
      for(std::size_t i=0; i < arg_DgDps[winIndex].size(); i++) 
	dgdp->ReplaceGlobalValue(offset, i, arg_DgDps[winIndex][i]);
    }
  }

  // distance btwn 1D, 2D or 3D points (params ordered as (x1,y1,z1,x2,y2,z2) )
  else if( fnName == "dist") { 
    if(nArgs == 2) 
      g[offset] = abs(arg_vals[0]-arg_vals[1]);
    else if(nArgs == 4) 
      g[offset] = sqrt( pow(arg_vals[0]-arg_vals[2],2) + pow(arg_vals[1]-arg_vals[3],2));
    else if(nArgs == 6) 
      g[offset] = sqrt( pow(arg_vals[0]-arg_vals[3],2) + 
			pow(arg_vals[1]-arg_vals[4],2) + pow(arg_vals[2]-arg_vals[5],2) );

    if(dgdp != Teuchos::null) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Error! No implementation of derivatives for distance function yet." << std::endl);
    }
  }

  // multiplicative scaling
  else if( fnName == "scale") {
    g[offset] = arg_vals[0] * arg_vals[1]; //set value

    if(dgdp != Teuchos::null) { //set derivative (muliplication rule)
      for(std::size_t i=0; i < nParameters; i++) { 
	// g = f1(p) * f2(p) ==> dgdpi = f1(p) * d(f2)/dpi + d(f1)/dpi * f2(p)
	dgdp->ReplaceGlobalValue(offset, i, arg_vals[0] * arg_DgDps[1][i] + arg_DgDps[0][i] * arg_vals[1]);
      }
    }
  }

  // multiplicative scaling
  else if( fnName == "divide") {
    g[offset] = arg_vals[0] / arg_vals[1]; //set value

    if(dgdp != Teuchos::null) { //set derivative (quotient rule)
      for(std::size_t i=0; i < nParameters; i++) { 
	// g = f1(p) / f2(p)  ==> dgdpi =  [ d(f1)/dpi * f2(p) - f1(p) * d(f2)/dpi ] / f2(p)^2
	dgdp->ReplaceGlobalValue(offset, i, (arg_DgDps[0][i] * arg_vals[1] - arg_vals[0] * arg_DgDps[1][i]) / pow(arg_vals[1],2) );
      }
    }
  }

  // no op (but can pass through multiple doubles)
  else if( fnName == "nop") {
    for(std::size_t i=0; i < nArgs; i++) {
      g[offset+i] = arg_vals[i]; //set value

      if(dgdp != Teuchos::null) { //set derivative
        for(std::size_t k=0; k < arg_DgDps[i].size(); k++) { //set derivative
#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
          typedef int GlobalIndex;
#else
          typedef long long GlobalIndex;
#endif
          dgdp->ReplaceGlobalValue(static_cast<GlobalIndex>(offset+i), k, arg_DgDps[i][k]);
        }
      }
    }
  }

  // sensitivity element: DgDp( SolverName, pIndex, gIndex )
  else if( fnName == "DgDp") {

    if(bSupportDpDg) {
      int pIndex = (int)arg_vals[0], gIndex = (int)arg_vals[1];
      Teuchos::RCP<Epetra_MultiVector> sub_dgdp = 
	(subSolvers.find(dgdpName)->second).responses_out->get_DgDp(0,0).getMultiVector(); // only use first g & p vectors

    
      //Note: this assumes vectors use a local map so [pIndex] element exists on all procs (later fix using import to local map)
      TEUCHOS_TEST_FOR_EXCEPTION(sub_dgdp->DistributedGlobal(), Teuchos::Exceptions::InvalidParameter,
				 "Error! sub-solvers's DgDp multivector is distributed.  No implementation for this yet."
				 << std::endl);

      g[offset] = (*((*sub_dgdp)(pIndex)))[gIndex]; 

      if(dgdp != Teuchos::null) { //set QCAD::Solver derivative to zero (no derivatives of derivatives)
	for(std::size_t k=0; k < nParameters; k++)
	  dgdp->ReplaceGlobalValue(offset,k, 0.0); 
      }
    }
    else g[offset] = 0.0; // just set response as zero if dgdp isn't supported
  }
 
  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			  "Unknown function " << fnName << " for QCAD solver response." << std::endl);
}






// Helper Functions

void QCAD::SolveModel(const QCAD::SolverSubSolver& ss)
{
  ss.model->evalModel( (*ss.params_in), (*ss.responses_out) );
}

void QCAD::SolveModel(const QCAD::SolverSubSolver& ss, 
		Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates)
{
  if(pInitialStates != NULL) 
    ss.app->getStateMgr().importStateData( *pInitialStates );

  ss.model->evalModel( (*ss.params_in), (*ss.responses_out) );

  pFinalStates = &(ss.app->getStateMgr().getStateArrays());
}

void QCAD::SolveModel(const QCAD::SolverSubSolver& ss, 
		Teuchos::RCP<Albany::EigendataStruct>& pInitialEData, 
		Teuchos::RCP<Albany::EigendataStruct>& pFinalEData)
{
  if(pInitialEData != Teuchos::null) 
    ss.app->getStateMgr().setEigenData(pInitialEData);

  ss.model->evalModel( (*ss.params_in), (*ss.responses_out) );

  pFinalEData = ss.app->getStateMgr().getEigenData();
}


void QCAD::SolveModel(const QCAD::SolverSubSolver& ss, 
		Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates,
		Teuchos::RCP<Albany::EigendataStruct>& pInitialEData, 
		Teuchos::RCP<Albany::EigendataStruct>& pFinalEData)
{
  if(pInitialStates != NULL) 
    ss.app->getStateMgr().importStateData( *pInitialStates );

  if(pInitialEData != Teuchos::null) 
    ss.app->getStateMgr().setEigenData(pInitialEData);

  ss.model->evalModel( (*ss.params_in), (*ss.responses_out) );

  pFinalStates = &(ss.app->getStateMgr().getStateArrays());
  pFinalEData = ss.app->getStateMgr().getEigenData();
}


void QCAD::CopyStateToContainer(Albany::StateArrays& src,
			  std::string stateNameToCopy,
			  std::vector<Intrepid::FieldContainer<RealType> >& dest)
{
  int numWorksets = src.size();
  std::vector<int> dims;

  //allocate destination container if necessary
  if(dest.size() != (unsigned int)numWorksets) {
    dest.resize(numWorksets);    
    for (int ws = 0; ws < numWorksets; ws++) {
      src[ws][stateNameToCopy].dimensions(dims);
      dest[ws].resize(dims);
    }
  }

  for (int ws = 0; ws < numWorksets; ws++)
  {
    src[ws][stateNameToCopy].dimensions(dims);
    TEUCHOS_TEST_FOR_EXCEPT( dims.size() != 2 );
    
    for(int cell=0; cell < dims[0]; cell++)
      for(int qp=0; qp < dims[1]; qp++)
        dest[ws](cell,qp) = src[ws][stateNameToCopy](cell,qp);
  }
}


//Note: state must be allocated already
void QCAD::CopyContainerToState(std::vector<Intrepid::FieldContainer<RealType> >& src,
			  Albany::StateArrays& dest,
			  std::string stateNameOfCopy)
{
  int numWorksets = src.size();
  std::vector<int> dims;

  for (int ws = 0; ws < numWorksets; ws++)
  {
    dest[ws][stateNameOfCopy].dimensions(dims);
    TEUCHOS_TEST_FOR_EXCEPT( dims.size() != 2 );
    
    for(int cell=0; cell < dims[0]; cell++) {
      for(int qp=0; qp < dims[1]; qp++) {
	TEUCHOS_TEST_FOR_EXCEPT( std::isnan(src[ws](cell,qp)) );
        dest[ws][stateNameOfCopy](cell,qp) = src[ws](cell,qp);
      }
    }
  }
}


//Note: assumes src and dest have allocated states of <stateNameToCopy>
void QCAD::CopyState(Albany::StateArrays& src,
	       Albany::StateArrays& dest,
	       std::string stateNameToCopy)
{
  int numWorksets = src.size();
  int totalSize;

  for (int ws = 0; ws < numWorksets; ws++)
  {
    totalSize = src[ws][stateNameToCopy].size();
    for(int i=0; i<totalSize; ++i)
      dest[ws][stateNameToCopy][i] = src[ws][stateNameToCopy][i];
  }
}


void QCAD::AddStateToState(Albany::StateArrays& src,
		     std::string srcStateNameToAdd, 
		     Albany::StateArrays& dest,
		     std::string destStateNameToAddTo)
{
  int totalSize, numWorksets = src.size();
  TEUCHOS_TEST_FOR_EXCEPT( numWorksets != (int)dest.size() );

  for (int ws = 0; ws < numWorksets; ws++)
  {
    totalSize = src[ws][srcStateNameToAdd].size();
    
    for(int i=0; i<totalSize; ++i)
      dest[ws][destStateNameToAddTo][i] += src[ws][srcStateNameToAdd][i];
  }
}


void QCAD::SubtractStateFromState(Albany::StateArrays& src, 
			    std::string srcStateNameToSubtract,
			    Albany::StateArrays& dest,
			    std::string destStateNameToSubtractFrom)
{
  int totalSize, numWorksets = src.size();
  TEUCHOS_TEST_FOR_EXCEPT( numWorksets != (int)dest.size() );

  for (int ws = 0; ws < numWorksets; ws++)
  {
    totalSize = src[ws][srcStateNameToSubtract].size();
    
    for(int i=0; i<totalSize; ++i)
      dest[ws][destStateNameToSubtractFrom][i] -= src[ws][srcStateNameToSubtract][i];
  }
}

double QCAD::getMaxDifference(Albany::StateArrays& states, 
		      std::vector<Intrepid::FieldContainer<RealType> >& prevState,
		      std::string stateName)
{
  double maxDiff = 0.0;
  int numWorksets = states.size();
  std::vector<int> dims;

  TEUCHOS_TEST_FOR_EXCEPT( ! (numWorksets == (int)prevState.size()) );

  for (int ws = 0; ws < numWorksets; ws++)
  {
    states[ws][stateName].dimensions(dims);
    TEUCHOS_TEST_FOR_EXCEPT( dims.size() != 2 );
    
    for(int cell=0; cell < dims[0]; cell++) 
    {
      for(int qp=0; qp < dims[1]; qp++) 
      {
        // std::cout << "prevState = " << prevState[ws](cell,qp) << std::endl;
        // std::cout << "currState = " << states[ws][stateName](cell,qp) << std::endl;
        if( fabs( states[ws][stateName](cell,qp) - prevState[ws](cell,qp) ) > maxDiff ) 
	  maxDiff = fabs( states[ws][stateName](cell,qp) - prevState[ws](cell,qp) );
      }
    }
  }
  return maxDiff;
}


void QCAD::ResetEigensolverShift(const Teuchos::RCP<EpetraExt::ModelEvaluator>& Solver, double newShift, 
			   Teuchos::RCP<Teuchos::ParameterList>& eigList) 
{
  Teuchos::RCP<Piro::Epetra::LOCASolver> pels = Teuchos::rcp_dynamic_cast<Piro::Epetra::LOCASolver>(Solver);
  TEUCHOS_TEST_FOR_EXCEPT(pels == Teuchos::null);

  Teuchos::RCP<LOCA::Stepper> stepper =  pels->getLOCAStepperNonConst();
  const Teuchos::ParameterList& oldEigList = stepper->getList()->sublist("LOCA").sublist(
							 "Stepper").sublist("Eigensolver");

  eigList = Teuchos::rcp(new Teuchos::ParameterList(oldEigList));
  eigList->set("Shift",newShift);

  //cout << " OLD Eigensolver list  " << oldEigList << std::endl;
  //cout << " NEW Eigensolver list  " << *eigList << std::endl;
  std::cout << "QCAD Solver setting eigensolver shift = " 
	    << std::setprecision(5) << newShift << std::endl;

  stepper->eigensolverReset(eigList);
}


double QCAD::GetEigensolverShift(const QCAD::SolverSubSolver& ss, double pcBelowMinPotential)
{
  int Ng = ss.responses_out->Ng();
  TEUCHOS_TEST_FOR_EXCEPT( Ng <= 0 );

  Teuchos::RCP<Epetra_Vector> gVector = ss.responses_out->get_g(0);

  //assume Field Value response that computes the minimum is the last "component" response function
  // comprising the aggregated response function that fills response vector 0.  A Field Value response
  // computes 5 doubles, and the value we're after is the first, so we want the element 5 from the end.
  int minPotentialResponseIndex = gVector->GlobalLength() - 5;
  double minVal = (*gVector)[minPotentialResponseIndex];

  //set shift to be slightly (5% of range) below minimum value
  //double shift = -(minVal - 0.05*(maxVal-minVal)); //minus sign b/c negative eigenvalue convention
  
  //double shift = -minVal*1.1;  // 10% below minimum value
  double shift = -minVal*(1.0 + pcBelowMinPotential/100.0);
  return shift;
}
  



// String functions
std::vector<std::string> QCAD::string_split(const std::string& s, char delim, bool bProtect)
{
  int last = 0;
  int parenLevel=0, bracketLevel=0, braceLevel=0;

  std::vector<std::string> ret(1);
  for(std::size_t i=0; i<s.size(); i++) {
    if(s[i] == delim && parenLevel==0 && bracketLevel==0 && braceLevel==0) {
      ret.push_back(""); 
      last++; 
    }
    else ret[last] += s[i];

    if(bProtect) {
      if(     s[i] == '(') parenLevel++;
      else if(s[i] == ')') parenLevel--;
      else if(s[i] == '[') bracketLevel++;
      else if(s[i] == ']') bracketLevel--;
      else if(s[i] == '{') braceLevel++;
      else if(s[i] == '}') braceLevel--;
    }
  }
  return ret;
}

std::string QCAD::string_remove_whitespace(const std::string& s)
{
  std::string ret;
  for(std::size_t i=0; i<s.size(); i++) {
    if(s[i] == ' ') continue;
    ret += s[i];
  }
  return ret;
}

//given s="MyFunction(arg1,arg2,arg3)", 
// returns vector { "MyFunction", "arg1", "arg2", "arg3" }
std::vector<std::string> QCAD::string_parse_function(const std::string& s)
{
  std::vector<std::string> ret;
  std::string fnName, fnArgString;
  std::size_t firstOpenParen, lastCloseParen;

  firstOpenParen = s.find_first_of('(');
  lastCloseParen = s.find_last_of(')');

  if(firstOpenParen == std::string::npos || lastCloseParen == std::string::npos) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Malformed function string: " << s << std::endl);
  }

  fnName = s.substr(0,firstOpenParen);
  fnArgString = s.substr(firstOpenParen+1,lastCloseParen-firstOpenParen-1);

  ret = QCAD::string_split(fnArgString,',',true);
  ret.insert(ret.begin(),fnName);  //place function name at beginning of vector returned
  return ret;
}

std::map<std::string,std::string> QCAD::string_parse_arrayref(const std::string& s)
{
  std::map<std::string,std::string> ret;
  std::string arName, indexString;
  std::size_t firstOpenBracket, lastCloseBracket;

  firstOpenBracket = s.find_first_of('[');
  lastCloseBracket = s.find_last_of(']');

  if(firstOpenBracket == std::string::npos || lastCloseBracket == std::string::npos) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Malformed array string: " << s << std::endl);
  }

  arName = s.substr(0,firstOpenBracket);
  indexString = s.substr(firstOpenBracket+1,lastCloseBracket-firstOpenBracket-1);

  ret["name"] = arName;
  ret["index"] = indexString;
  return ret;
}

std::vector<int> QCAD::string_expand_compoundindex(const std::string& indexStr, int min_index, int max_index)
{
  std::vector<int> ret;
  std::vector<std::string> simpleRanges = QCAD::string_split(indexStr,',',true);
  std::vector<std::string>::const_iterator it;

  for(it = simpleRanges.begin(); it != simpleRanges.end(); it++) {
    std::vector<std::string> endpts = QCAD::string_split( (*it), ':', true);
    if(endpts.size() == 1)
      ret.push_back( atoi(endpts[0].c_str()) );
    else if(endpts.size() == 2) {
      int a=min_index ,b=max_index;
      if(endpts[0] != "") a = atoi(endpts[0].c_str());
      if(endpts[1] != "") b = atoi(endpts[1].c_str());

      TEUCHOS_TEST_FOR_EXCEPTION(a < min_index || a > max_index || b < min_index || b > max_index,
				 Teuchos::Exceptions::InvalidParameter, "Index '"<< indexStr 
				 << "' is out of bounds (min="<<min_index<<", max="<<max_index<<")" << std::endl);

      for(int i=a; i<b; i++) ret.push_back(i);
    }
    else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		       "Malformed array index: " << indexStr << std::endl);
  }
  return ret;
}

