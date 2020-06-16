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

template <class Disc>
Albany::BlockedDiscretization<Disc>::BlockedDiscretization(
    const Teuchos::RCP<Teuchos::ParameterList>&    discParams_,
    Teuchos::RCP<Albany::AbstractSTKMeshStruct>&   stkMeshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>&        comm_,
    const Teuchos::RCP<Albany::RigidBodyModes>&    rigidBodyModes_,
    const std::map<int, std::vector<std::string>>& sideSetEquations_){

	Traits = Teuchos::rcp(new traits_type(discParams_, stkMeshStruct_, comm_,
		rigidBodyModes_, sideSetEquations_));
}

template <class Disc>
Albany::BlockedDiscretization<Disc>::~BlockedDiscretization()
{

	// explicitly destroy the object the pointer points at

    Traits = Teuchos::null;

}

template <class Disc>
void
Albany::BlockedDiscretization<Disc>::printConnectivity() const
{

	Traits->printConnectivity();

}

template <class Disc>
Teuchos::RCP<const Thyra_VectorSpace>
Albany::BlockedDiscretization<Disc>::getVectorSpace(const std::string& field_name) const
{
  return Traits->getVectorSpace();
}

template <class Disc>
Teuchos::RCP<const Thyra_VectorSpace>
Albany::BlockedDiscretization<Disc>::getNodeVectorSpace(const std::string& field_name) const
{
  return Traits->getNodeVectorSpace();
}

template <class Disc>
Teuchos::RCP<const Thyra_VectorSpace>
Albany::BlockedDiscretization<Disc>::getOverlapVectorSpace(const std::string& field_name) const
{
  return Traits->getOverlapVectorSpace();
}

template <class Disc>
Teuchos::RCP<const Thyra_VectorSpace>
Albany::BlockedDiscretization<Disc>::getOverlapNodeVectorSpace(
    const std::string& field_name) const
{
  return Traits->getOverlapNodeVectorSpace();
}

template <class Disc>
void
Albany::BlockedDiscretization<Disc>::printCoords() const
{
	Traits->printCoords();
}

template <class Disc>
const Teuchos::ArrayRCP<double>&
Albany::BlockedDiscretization<Disc>::getCoordinates() const
{

  return Traits->getCoordinates();

}

// These methods were added to support mesh adaptation, which is currently
// limited to PUMIDiscretization.
template <class Disc>
void
Albany::BlockedDiscretization<Disc>::setCoordinates(
    const Teuchos::ArrayRCP<const double>& /* c */)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::logic_error,
      "BlockedDiscretization::setCoordinates is not implemented.");
}

// The function transformMesh() maps a unit cube domain by applying a
// transformation to the mesh.
template <class Disc>
void
Albany::BlockedDiscretization<Disc>::transformMesh()
{

	Traits->transformMesh();

}

template <class Disc>
void
Albany::BlockedDiscretization<Disc>::writeSolutionMV(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
    const double             time,
    const bool               overlapped)
{
  Traits->writeSolutionMV(soln, solution_dxdp, time, overlapped);
}

template <class Disc>
GO
Albany::BlockedDiscretization<Disc>::gid(const stk::mesh::Entity node) const
{
  return Traits->gid(node);
}

template <class Disc>
int
Albany::BlockedDiscretization<Disc>::getOwnedDOF(const int inode, const int eq) const
{
	return getOwnedDOF(inode, eq);
}

template <class Disc>
int
Albany::BlockedDiscretization<Disc>::getOverlapDOF(const int inode, const int eq) const
{
	return Traits->getOverlapDOF(inode, eq);
}

template <class Disc>
GO
Albany::BlockedDiscretization<Disc>::getGlobalDOF(const GO inode, const int eq) const
{
	return Traits->getGlobalDOF(inode, eq);
}

template <class Disc>
void
Albany::BlockedDiscretization<Disc>::reNameExodusOutput(std::string& filename)
{
	Traits->reNameExodusOutput(filename);
}

template <class Disc>
void
Albany::BlockedDiscretization<Disc>::updateMesh()
{
	Traits->updateMesh();
}
