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


#include <iostream>

#include "Albany_Point0DSTKMeshStruct.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/fem/FEMHelpers.hpp>

#ifdef ALBANY_IOSS
#include <stk_io/IossBridge.hpp>
#endif
#include "Albany_Utils.hpp"

Albany::Point0DSTKMeshStruct::Point0DSTKMeshStruct(
		  const Teuchos::RCP<Teuchos::ParameterList>& params) :
  GenericSTKMeshStruct(params,1) // Really 0D, but STK prefers to think of it as 1D
{

  params->validateParameters(*getValidDiscretizationParameters(),0);
  std::vector<std::string> nsNames;
  this->DeclareParts(nsNames);
  stk::mesh::fem::set_cell_topology< shards::Particle >(*partVec[0]);

  int cub = params->get("Cubature Degree",3);
  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
  this->meshSpecs = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub, nsNames));
}

void
Albany::Point0DSTKMeshStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_, const unsigned int nstates_,
                  const unsigned int worksetSize)
{
  cout << "XXX Point0DSTKMeshStruct::setFieldAndBulkData " << endl;

  const int nelem = 1;

  // Distribute the elements equally among processors
  Teuchos::RCP<Epetra_Map> elem_map = Teuchos::rcp(new Epetra_Map(nelem, 0, *comm));


  this->SetupFieldData(comm, neq_, nstates_, worksetSize);
  
  metaData->commit();

  // Finished with metaData, now work on bulk data

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh
  std::vector<stk::mesh::Part*> noPartVec;
  std::vector<stk::mesh::Part*> singlePartVec(1);

    singlePartVec[0] = partVec[0];

    stk::mesh::Entity& pt  = bulkData->declare_entity(metaData->element_rank(), 1, singlePartVec);
    stk::mesh::Entity& node = bulkData->declare_entity(metaData->node_rank(), 1, noPartVec);
    bulkData->declare_relation(pt, node, 0);

  bulkData->modification_end();
  useElementAsTopRank = true;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::Point0DSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("ValidSTK0D_DiscParams");

  return validPL;
}

