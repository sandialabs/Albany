//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <limits>

#include <Albany_ThyraUtils.hpp>
#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include "BlockedDiscretization.hpp"

//Assume we only have one block right now
Albany::BlockedDiscretization::BlockedDiscretization(
    const Teuchos::RCP<Teuchos::ParameterList>&    discParams_,
    Teuchos::RCP<Albany::AbstractSTKMeshStruct>&   stkMeshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>&        comm_,
    const Teuchos::RCP<Albany::RigidBodyModes>&    rigidBodyModes_,
    const std::map<int, std::vector<std::string>>& sideSetEquations_){

    BlockDiscretization.resize(1);

	BlockDiscretization[0] = Teuchos::rcp(new BlckDisc(discParams_, stkMeshStruct_, comm_,
		rigidBodyModes_, sideSetEquations_));
}

Albany::BlockedDiscretization::~BlockedDiscretization()
{

	// explicitly destroy the object the pointer points at

    BlockDiscretization[0] = Teuchos::null;

}

void
Albany::BlockedDiscretization::printConnectivity() const
{

	BlockDiscretization[0]->printConnectivity();

}

Teuchos::RCP<const Thyra_VectorSpace>
Albany::BlockedDiscretization::getVectorSpace(const std::string& field_name) const
{
  return BlockDiscretization[0]->getVectorSpace();
}

Teuchos::RCP<const Thyra_VectorSpace>
Albany::BlockedDiscretization::getNodeVectorSpace(const std::string& field_name) const
{
  return BlockDiscretization[0]->getNodeVectorSpace();
}

Teuchos::RCP<const Thyra_VectorSpace>
Albany::BlockedDiscretization::getOverlapVectorSpace(const std::string& field_name) const
{
  return BlockDiscretization[0]->getOverlapVectorSpace();
}

Teuchos::RCP<const Thyra_VectorSpace>
Albany::BlockedDiscretization::getOverlapNodeVectorSpace(
    const std::string& field_name) const
{
  return BlockDiscretization[0]->getOverlapNodeVectorSpace();
}

void
Albany::BlockedDiscretization::printCoords() const
{
	BlockDiscretization[0]->printCoords();
}

const Teuchos::ArrayRCP<double>&
Albany::BlockedDiscretization::getCoordinates() const
{

  return BlockDiscretization[0]->getCoordinates();

}

// These methods were added to support mesh adaptation, which is currently
// limited to PUMIDiscretization.
void
Albany::BlockedDiscretization::setCoordinates(
    const Teuchos::ArrayRCP<const double>& /* c */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "BlockedDiscretization::setCoordinates is not implemented.");
}

// The function transformMesh() maps a unit cube domain by applying a
// transformation to the mesh.
void
Albany::BlockedDiscretization::transformMesh()
{

	BlockDiscretization[0]->transformMesh();

}

void
Albany::BlockedDiscretization::writeSolutionMV(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
    const double             time,
    const bool               overlapped)
{
  BlockDiscretization[0]->writeSolutionMV(soln, solution_dxdp, time, overlapped);
}

GO
Albany::BlockedDiscretization::gid(const stk::mesh::Entity node) const
{
  return BlockDiscretization[0]->gid(node);
}

int
Albany::BlockedDiscretization::getOwnedDOF(const int inode, const int eq) const
{
	return getOwnedDOF(inode, eq);
}

int
Albany::BlockedDiscretization::getOverlapDOF(const int inode, const int eq) const
{
	return BlockDiscretization[0]->getOverlapDOF(inode, eq);
}

GO
Albany::BlockedDiscretization::getGlobalDOF(const GO inode, const int eq) const
{
	return BlockDiscretization[0]->getGlobalDOF(inode, eq);
}

void
Albany::BlockedDiscretization::reNameExodusOutput(std::string& filename)
{
	BlockDiscretization[0]->reNameExodusOutput(filename);
}

void
Albany::BlockedDiscretization::updateMesh()
{
	BlockDiscretization[0]->updateMesh();
}
