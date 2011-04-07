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
		  const Teuchos::RCP<const Epetra_Comm>& comm,
		  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_, const unsigned int nstates_) :
  GenericSTKMeshStruct(comm)
{

  params->validateParameters(*getValidDiscretizationParameters(),0);
  int numDim_ = 1;

  const int nelem = 1;

  // Distribute the elements equally among processors
  Teuchos::RCP<Epetra_Map> elem_map = Teuchos::rcp(new Epetra_Map(nelem, 0, *comm));

  std::vector<std::string> nsNames;

  this->SetupMetaData(params, neq_, nstates_, numDim_);
  this->DeclareParts(nsNames);
  
  stk::mesh::fem::set_cell_topology_new< shards::Particle >(*partVec[0]);
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

