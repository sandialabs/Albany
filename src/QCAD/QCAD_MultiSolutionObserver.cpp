//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"
#include "Epetra_Export.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"

#include "Albany_StateInfoStruct.hpp"
#include "QCAD_MultiSolutionObserver.hpp"

//For creating discretiation object
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_AbstractFieldContainer.hpp"
#include "Piro_NullSpaceUtils.hpp"



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OBSERVER
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

QCAD::MultiSolution_Observer::MultiSolution_Observer(const Teuchos::RCP<Albany::Application>& app,
						     const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  rootParams = Teuchos::createParameterList("Multi Solution Observer Discretization Parameters");
  Teuchos::ParameterList& discList = rootParams->sublist("Discretization", false);
  if(params->isSublist("Discretization"))
    discList.setParameters(params->sublist("Discretization")); //copy discretization sublist of app

  apps.push_back(app);
}

QCAD::MultiSolution_Observer::MultiSolution_Observer(const Teuchos::RCP<Albany::Application>& app1,
						     const Teuchos::RCP<Albany::Application>& app2,
						     const Teuchos::RCP<Teuchos::ParameterList>& params)
{
  rootParams = Teuchos::createParameterList("Multi Solution Observer Discretization Parameters");
  Teuchos::ParameterList& discList = rootParams->sublist("Discretization", false);
  if(params->isSublist("Discretization"))
    discList.setParameters(params->sublist("Discretization")); //copy discretization sublist of app

  apps.push_back(app1);
  apps.push_back(app2);
}


void QCAD::MultiSolution_Observer::observeSolution(const Epetra_Vector& solution, const std::string& solutionLabel,
						   Teuchos::RCP<Albany::EigendataStruct> eigenData,
						   double stamp)
{
  if(apps.size() == 0) return;

  int nAppSolns = 1;  // one app solution (maybe more in overloads of observeSolution)
  
  // Step 1: determine the map for the full solution vector
  int nDiscMapCopies = nAppSolns;  
  if(eigenData != Teuchos::null) nDiscMapCopies += eigenData->eigenvalueRe->size();

  // Create combined vector map & full solution vector
  Teuchos::RCP<const Epetra_Comm> comm = apps[0]->getComm();
  Teuchos::RCP<const Epetra_Map> disc_map = apps[0]->getDiscretization()->getMap();
  Teuchos::RCP<const Epetra_Map> disc_overlap_map = apps[0]->getDiscretization()->getOverlapMap();
  Teuchos::RCP<Epetra_Map> combinedMap = QCAD::CreateCombinedMap(disc_map, nDiscMapCopies, 0, comm);
  
  Teuchos::RCP<Epetra_Vector> fullSoln = Teuchos::rcp(new Epetra_Vector(*combinedMap));
  Teuchos::RCP<Epetra_MultiVector> parts; 
  Teuchos::RCP<Epetra_Vector> dummy;
  
  // Separate full solution vector and fill in parts individually
  QCAD::separateCombinedVector(disc_map, nDiscMapCopies, 0, comm, fullSoln, parts, dummy);

    // Trivial Exporter: non-overlapped to non-overlapped data (just copies)
  Teuchos::RCP<Epetra_Export> trivial_exporter = Teuchos::rcp(new Epetra_Export(*disc_map, *disc_map));
  (*parts)(0)->Export( solution, *trivial_exporter, Insert); // assume solution vector is non-overlapped

    // Overlap Exporter: overlapped to non-overlapped data 
  Teuchos::RCP<Epetra_Export> overlap_exporter = Teuchos::rcp(new Epetra_Export(*disc_overlap_map, *disc_map));
  
  int nEigenvals;
  if(eigenData != Teuchos::null) { // transfer real parts of eigenvectors (note: they're stored in overlapped distribution)
    nEigenvals = eigenData->eigenvalueRe->size();
    for(int k=0; k < nEigenvals; k++) 
      (*parts)(nAppSolns+k)->Export( *((*(eigenData->eigenvectorRe))(k)), *overlap_exporter, Insert);
  }
  else nEigenvals = 0;



  //Build a new discretization object that expects combined_map solution vectors and
  // includes space for all app states.

  std::vector<std::string> solnVecComps( 2*nDiscMapCopies );  
  solnVecComps[0] = solutionLabel; solnVecComps[1] = "S";
  for(int i=0; i<nEigenvals; i++) {
    std::ostringstream ss1; ss1 << "Eigenvector" << i;
    solnVecComps[ 2*(i+nAppSolns) ] = ss1.str(); 
    solnVecComps[ 2*(i+nAppSolns)+1 ] = "S";
  }

  Teuchos::ParameterList& discList = rootParams->sublist("Discretization", true);
  discList.set("Solution Vector Components", Teuchos::Array<std::string>(solnVecComps));
  discList.set("Interleaved Ordering", false); //combined vector is concatenated, not "interleaved"

    // Create discretization factory
  Albany::DiscretizationFactory discFactory(rootParams, comm);

    // Get mesh specification object: worksetSize, cell topology, etc
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs =
    discFactory.createMeshSpecs();

  Albany::AbstractFieldContainer::FieldContainerRequirements requirements; //empty
  Teuchos::RCP<Albany::StateInfoStruct> stateInfo = apps[0]->getStateMgr().getStateInfoStruct(); //just use apps[0]'s states for now

    //Create a discretization object based on the one in apps[0]
  int neq = nDiscMapCopies;
  Teuchos::RCP<Piro::MLRigidBodyModes> rigidBodyModes(Teuchos::rcp(new Piro::MLRigidBodyModes(neq)));
  Teuchos::RCP<Albany::AbstractDiscretization> my_disc = discFactory.createDiscretization(neq, stateInfo,requirements,rigidBodyModes);  

    //Copy in states from apps (just apps[0] for now)
  Albany::StateArrays& my_states = my_disc->getStateArrays();
  Albany::StateArrays& appStates = apps[0]->getDiscretization()->getStateArrays();
  Teuchos::RCP<Albany::StateInfoStruct> appStateInfo = apps[0]->getStateMgr().getStateInfoStruct();
  QCAD::CopyAllStates(appStates, my_states, appStateInfo);

    //Finally, write out the solution using our the created discretization object
  my_disc->writeSolution(*fullSoln, stamp, /*overlapped =*/ false);
}



/*void observeSolution(
       const Epetra_Vector& solution, double time_or_param_val)
{
  Teuchos::RCP<const Epetra_Vector> soln_poisson, soln_eigenvals_dist;
  Teuchos::RCP<const Epetra_MultiVector> soln_schrodinger;

  psModel_->separateCombinedVector(Teuchos::rcp(&solution, false), 
				   soln_poisson, soln_schrodinger, soln_eigenvals_dist);

  int nEigenvals = soln_eigenvals_dist->Map().NumGlobalElements();
  Teuchos::RCP<Albany::Application> poisson_app = psModel_->getPoissonApp();      
  Teuchos::RCP<Albany::Application> schrodinger_app = psModel_->getSchrodingerApp();      

  // Evaluate state field managers
  poisson_app->evaluateStateFieldManager(time_or_param_val, NULL, *soln_poisson);
  for(int i=0; i<nEigenvals; i++)
    schrodinger_app->evaluateStateFieldManager(time_or_param_val, NULL, *((*soln_schrodinger)(i)) );

  // This must come at the end since it renames the New state 
  // as the Old state in preparation for the next step
  poisson_app->getStateMgr().updateStates();
  schrodinger_app->getStateMgr().updateStates();

  Epetra_Vector *poisson_ovlp_solution = poisson_app->getAdaptSolMgr()->getOverlapSolution(*soln_poisson);
  poisson_app->getDiscretization()->writeSolution(*poisson_ovlp_solution, time_or_param_val, true); // soln is overlapped

  for(int i=0; i<nEigenvals; i++) {
    Epetra_Vector *schrodinger_ovlp_solution = schrodinger_app->getAdaptSolMgr()->getOverlapSolution(*((*soln_schrodinger)(i)));
    schrodinger_app->getDiscretization()->writeSolution(*schrodinger_ovlp_solution, time_or_param_val + i*0.1, true); // soln is overlapped
  }

  soln_eigenvals_dist->Print(std::cout << "Coupled PS Solution Eigenvalues:" << std::endl);

  // States: copy states from Poission app's discretization object into psModel's object before writing solution
  Albany::StateArrays& psDiscStates = psModel_->getDiscretization()->getStateArrays();
  Albany::StateArrays& psPoissonStates = poisson_app->getDiscretization()->getStateArrays();
  Teuchos::RCP<Albany::StateInfoStruct> stateInfo = poisson_app->getStateMgr().getStateInfoStruct();
  CopyAllStates(psPoissonStates, psDiscStates, stateInfo);

  
  //Test: use discretization built by coupled poisson-schrodinger model, which has separated solution vector specified in input file
  psModel_->getDiscretization()->writeSolution(solution, time_or_param_val, false); // soln is non-overlapped
}
*/



//////////////////////////////////////////////////////////////////////////////////////////////
//    Utility functions
//////////////////////////////////////////////////////////////////////////////////////////////


//Note: assumes dest contains all the allocated states of src
void QCAD::CopyAllStates(Albany::StateArrays& state_arrays,
			 Albany::StateArrays& dest_arrays,
			 const Teuchos::RCP<const Albany::StateInfoStruct>& stateInfo)
{
  Albany::StateArrayVec& src = state_arrays.elemStateArrays;
  Albany::StateArrayVec& dest = dest_arrays.elemStateArrays;
  int numWorksets = src.size();
  int totalSize;

  for(std::size_t st=0; st < stateInfo->size(); st++) {
    const Teuchos::RCP<Albany::StateStruct>& ss = (*stateInfo)[st];
    std::string stateNameToCopy = ss->name;
    //TODO: check ss to make sure this is a scalar field that copies correctly below
    
    for (int ws = 0; ws < numWorksets; ws++) {
      totalSize = src[ws][stateNameToCopy].size();
      for(int i=0; i<totalSize; ++i)
	dest[ws][stateNameToCopy][i] = src[ws][stateNameToCopy][i];
    }
  }
}

Teuchos::RCP<Epetra_Map> QCAD::CreateCombinedMap(Teuchos::RCP<const Epetra_Map> disc_map,
						 int numCopiesOfDiscMap, int numAdditionalElements,
						 const Teuchos::RCP<const Epetra_Comm>& comm)
{
  //  Create a map which is the product of <nCopiesOfDiscMap> disc_maps + numAdditionalElements extra elements
  //  in such a way that the elements for each disc_map are contiguous in index space (so that we can easily get
  //  Epetra vector views to them separately)

  int myRank = comm->MyPID();
  int nProcs = comm->NumProc();
  int nExtra = numAdditionalElements % nProcs;

  int my_nAdditional = (numAdditionalElements / nProcs) + ((myRank < nExtra) ? 1 : 0);
  int my_scalar_offset = myRank * (numAdditionalElements / nProcs) + (myRank < nExtra) ? myRank : nExtra;
  int my_nElements = disc_map->NumMyElements() * numCopiesOfDiscMap + my_nAdditional;
  std::vector<int> my_global_elements(my_nElements);  //global element indices for this processor

  int disc_nGlobalElements = disc_map->NumGlobalElements();
  int disc_nMyElements = disc_map->NumMyElements();
  std::vector<int> disc_global_elements(disc_nMyElements);
  disc_map->MyGlobalElements(&disc_global_elements[0]);
  
  for(int k=0; k<numCopiesOfDiscMap; k++) {
    for(int l=0; l<disc_nMyElements; l++) {
      my_global_elements[k*disc_nMyElements + l] = k*disc_nGlobalElements + disc_global_elements[l];
    }
  }

  for(int l=0; l < my_nAdditional; l++) {
    my_global_elements[numCopiesOfDiscMap*disc_nMyElements + l] = numCopiesOfDiscMap*disc_nGlobalElements + my_scalar_offset + l;
  }
  
  int global_nElements = numCopiesOfDiscMap*disc_nGlobalElements + numAdditionalElements;
  return Teuchos::rcp(new Epetra_Map(global_nElements, my_nElements, &my_global_elements[0], 0, *comm));
}


void QCAD::separateCombinedVector(Teuchos::RCP<const Epetra_Map> disc_map,
				  int numCopiesOfDiscMap, int numAdditionalElements,
				  const Teuchos::RCP<const Epetra_Comm>& comm,
				  const Teuchos::RCP<Epetra_Vector>& combinedVector,
				  Teuchos::RCP<Epetra_MultiVector>& disc_parts,
				  Teuchos::RCP<Epetra_Vector>& additional_part)
{
  double* data;
  int disc_nMyElements = disc_map->NumMyElements();
  int my_nAdditionalEls = combinedVector->Map().NumMyElements() - disc_nMyElements * numCopiesOfDiscMap;
  Epetra_Map dist_additionalEl_map(numAdditionalElements, my_nAdditionalEls, 0, *comm);

  if(combinedVector->ExtractView(&data) != 0)
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Error!  QCAD::separateCombinedVector cannot extract vector views");

  disc_parts = Teuchos::rcp(new Epetra_MultiVector(::View, *disc_map, &data[0], disc_nMyElements, numCopiesOfDiscMap));
  additional_part = Teuchos::rcp(new Epetra_Vector(::View, dist_additionalEl_map, &data[numCopiesOfDiscMap*disc_nMyElements]));
  return;
}
