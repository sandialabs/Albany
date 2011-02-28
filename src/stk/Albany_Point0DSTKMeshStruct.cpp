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

#include <stk_mesh/fem/FieldDeclarations.hpp>
#include <stk_mesh/fem/TopologyHelpers.hpp>
#include <stk_mesh/fem/EntityRanks.hpp>

#ifdef ALBANY_IOSS
#include <stk_io/IossBridge.hpp>
#endif
#include "Albany_Utils.hpp"

enum { field_data_chunk_size = 1001 };


Albany::Point0DSTKMeshStruct::Point0DSTKMeshStruct(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
		  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_, const unsigned int nstates_) 
{

  params->validateParameters(*getValidDiscretizationParameters(),0);

  numDim = 0;
  neq = neq_;
  nstates = nstates_;

  const int nelem = 1;

  // Distribute the elements equally among processors
  Teuchos::RCP<Epetra_Map> elem_map = Teuchos::rcp(new Epetra_Map(nelem, 0, *comm));
  //int numMyElements = elem_map->NumMyElements();

  //Start STK stuff, from UseCase_2 constructor
  metaData = new stk::mesh::MetaData(stk::mesh::fem_entity_rank_names() );
  bulkData = new stk::mesh::BulkData(*metaData , Albany::getMpiCommFromEpetraComm(*comm), field_data_chunk_size );
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  solution_field = & metaData->declare_field< VectorFieldType >( "solution" );
  residual_field = & metaData->declare_field< VectorFieldType >( "residual" );
  state_field = & metaData->declare_field< VectorFieldType >( "state" );

  partVec[0] = &  metaData->declare_part( "Block_1", stk::mesh::Element );

  stk::mesh::set_cell_topology< shards::Node >(*partVec[0]);
  stk::mesh::put_field( *coordinates_field , stk::mesh::Node , metaData->universal_part() , numDim );
  stk::mesh::put_field( *solution_field , stk::mesh::Node , metaData->universal_part() , neq );
  stk::mesh::put_field( *residual_field , stk::mesh::Node , metaData->universal_part() , neq );
  if (nstates>0) stk::mesh::put_field( *state_field , stk::mesh::Element , metaData->universal_part() , nstates );


#ifdef ALBANY_IOSS
  stk::io::put_io_part_attribute(*partVec[0]);
  stk::io::set_field_role(*coordinates_field, Ioss::Field::ATTRIBUTE);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(*residual_field, Ioss::Field::TRANSIENT);
  if (nstates>0) stk::io::set_field_role(*state_field, Ioss::Field::TRANSIENT);
#endif
  metaData->commit();

  // Finished with metaData, now work on bulk data

  // STK
  bulkData->modification_begin(); // Begin modifying the mesh
  std::vector<stk::mesh::Part*> noPartVec;
  std::vector<stk::mesh::Part*> singlePartVec(1);

    singlePartVec[0] = partVec[0];

    stk::mesh::Entity& pt  = bulkData->declare_entity(stk::mesh::Element, 1, singlePartVec);
    stk::mesh::Entity& node = bulkData->declare_entity(stk::mesh::Node, 1, noPartVec);
    bulkData->declare_relation(pt, node, 0);

  bulkData->modification_end();
  useElementAsTopRank = true;
}

Albany::Point0DSTKMeshStruct::~Point0DSTKMeshStruct()
{ 
  delete metaData;
  delete bulkData;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::Point0DSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("ValidSTK0D_DiscParams"));;
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");

  return validPL;
}

