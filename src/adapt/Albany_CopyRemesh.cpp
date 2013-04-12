//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_CopyRemesh.hpp"

#include "Teuchos_TimeMonitor.hpp"

typedef stk::mesh::Entity Entity;
typedef stk::mesh::EntityRank EntityRank;
typedef stk::mesh::RelationIdentifier EdgeId;
typedef stk::mesh::EntityKey EntityKey;

Albany::CopyRemesh::
CopyRemesh(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_) :
    Albany::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
    remeshFileIndex(1)
{

  disc = StateMgr.getDiscretization();

	stk_discretization = static_cast<Albany::STKDiscretization *>(disc.get());

	stkMeshStruct = stk_discretization->getSTKMeshStruct();

	bulkData = stkMeshStruct->bulkData;
	metaData = stkMeshStruct->metaData;

	// The entity ranks
	nodeRank = metaData->NODE_RANK;
	edgeRank = metaData->EDGE_RANK;
	faceRank = metaData->FACE_RANK;
	elementRank = metaData->element_rank();

	numDim = stkMeshStruct->numDim;

  // Save the initial output file name
  baseExoFileName = stkMeshStruct->exoOutFile;

}

Albany::CopyRemesh::
~CopyRemesh()
{
}

bool
Albany::CopyRemesh::queryAdaptationCriteria(){

  int remesh_iter = params->get<int>("Remesh Step Number");

   if(iter == remesh_iter)
     return true;

  return false; 

}

bool
Albany::CopyRemesh::adaptMesh(){

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "Adapting mesh using Albany::CopyRemesh method      " << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

  // Save the current results and close the exodus file

  // Create a remeshed output file naming convention by adding the remeshFileIndex ahead of the period

  std::ostringstream ss;
  std::string str = baseExoFileName;
  ss << "_" << remeshFileIndex << ".";
  str.replace(str.find('.'), 1, ss.str());

  std::cout << "Remeshing: renaming output file to - " << str << endl;

  // Open the new exodus file for results
  stk_discretization->reNameExodusOutput(str);

  remeshFileIndex++;

  // do remeshing right here if we were doing any...

  // Throw away all the Albany data structures and re-build them from the mesh

  stk_discretization->updateMesh();

  return true;

}

//! Transfer solution between meshes.
void
Albany::CopyRemesh::
solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution){

   TEUCHOS_TEST_FOR_EXCEPT( oldSolution.MyLength() != newSolution.MyLength());

    newSolution = oldSolution;

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::CopyRemesh::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidCopyRemeshParameters");

/*
  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");
*/

  validPL->set<int>("Remesh Step Number", 1, "Iteration step at which to remesh the problem");

  return validPL;
}


