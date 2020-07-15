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

namespace Albany {

//Assume we only have one block right now
BlockedDiscretization::BlockedDiscretization(
    const Teuchos::RCP<Teuchos::ParameterList>&    discParams_,
    Teuchos::RCP<AbstractSTKMeshStruct>&   stkMeshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>&        comm_,
    const Teuchos::RCP<RigidBodyModes>&    rigidBodyModes_,
    const std::map<int, std::vector<std::string>>& sideSetEquations_)
{
  m_blocks.resize(1);

	m_blocks[0] = Teuchos::rcp(new disc_type(discParams_, stkMeshStruct_, comm_,
		rigidBodyModes_, sideSetEquations_));
}

void
BlockedDiscretization::printConnectivity() const
{
	m_blocks[0]->printConnectivity();
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedDiscretization::getVectorSpace(const std::string& field_name) const
{
  return m_blocks[0]->getVectorSpace(field_name);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedDiscretization::getNodeVectorSpace(const std::string& field_name) const
{
  return m_blocks[0]->getNodeVectorSpace(field_name);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedDiscretization::getOverlapVectorSpace(const std::string& field_name) const
{
  return m_blocks[0]->getOverlapVectorSpace(field_name);
}

Teuchos::RCP<const Thyra_VectorSpace>
BlockedDiscretization::getOverlapNodeVectorSpace(
    const std::string& field_name) const
{
  return m_blocks[0]->getOverlapNodeVectorSpace(field_name);
}

void
BlockedDiscretization::printCoords() const
{
	m_blocks[0]->printCoords();
}

const Teuchos::ArrayRCP<double>&
BlockedDiscretization::getCoordinates() const
{
  return m_blocks[0]->getCoordinates();
}

// These methods were added to support mesh adaptation, which is currently
// limited to PUMIDiscretization.
void
BlockedDiscretization::setCoordinates(
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
BlockedDiscretization::transformMesh()
{
	m_blocks[0]->transformMesh();
}

void
BlockedDiscretization::reNameExodusOutput(std::string& filename)
{
	m_blocks[0]->reNameExodusOutput(filename);
}

void
BlockedDiscretization::updateMesh()
{
	m_blocks[0]->updateMesh();
}

} // namespace Albany
