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

#ifdef ALBANY_IOSS

#include <iostream>

#include "Albany_IossSTKMeshStruct.hpp"
#include "Teuchos_VerboseObject.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/fem/FieldDeclarations.hpp>
#include <stk_mesh/fem/TopologyHelpers.hpp>
#include <stk_mesh/fem/EntityRanks.hpp>

#include <stk_io/util/UseCase_mesh.hpp>
#include <Ionit_Initializer.h>

#include <stk_io/IossBridge.hpp>

enum { field_data_chunk_size = 1001 };


Albany::IossSTKMeshStruct::IossSTKMeshStruct(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_) :
  periodic(params->get("Periodic BC", false))
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());


  params->validateParameters(*getValidDiscretizationParameters(),0);
  neq = neq_;

  cubatureDegree = params->get("Cubature Degree", 3);
  int index = params->get("Restart Index",-1); // Default to no restart

  metaData = new stk::mesh::MetaData(stk::mesh::fem_entity_rank_names() );
  bulkData = new stk::mesh::BulkData(*metaData , MPI_COMM_WORLD , field_data_chunk_size );
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  solution_field = & metaData->declare_field< VectorFieldType >( "solution" );
  stk::mesh::put_field( *solution_field , stk::mesh::Node , metaData->universal_part() , neq );
  stk::io::set_field_role(*coordinates_field, Ioss::Field::ATTRIBUTE);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);

  stk::io::util::MeshData* mesh_data = new stk::io::util::MeshData();

  Ioss::Init::Initializer io;
  stk::io::util::create_input_mesh("exodusii",
                             params->get<string>("Exodus Input File Name"),
                             "", MPI_COMM_WORLD, 
                             *metaData, *mesh_data, false); 

  *out << "AGS_IOSS: Loading STKMesh from exodus file  " << endl;

  partVec[0] = & metaData->universal_part();
  stk::io::put_io_part_attribute(*partVec[0]);

  metaData->commit();

  stk::io::util::populate_bulk_data(*bulkData, *mesh_data, "exodusii", index);
  bulkData->modification_end();


  // Set node sets
  const stk::mesh::PartVector & all_parts = metaData->get_parts();
  int numVerts;
  int nsid=0;
  for (stk::mesh::PartVector::const_iterator i = all_parts.begin();
       i != all_parts.end(); ++i) {

    stk::mesh::Part * const part = *i ;


    switch( part->primary_entity_rank() ) {
      case stk::mesh::Element:
          *out << "IOSS-STK: Element part found " << endl;
          // Since Cubit likes to define numDim=3 always, use vertex
          // count on top element block to figure out quad vs hex.
          numVerts = stk::mesh::get_cell_topology(*part)->vertex_count;
          if (numVerts==4) numDim=2;
          else if (numVerts==8) numDim=3;
          else TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                 std::endl << "Error!  IossSTKMeshStruct:  " <<
                 "Invalid vertex count from exodus mesh: " << numVerts << std::endl);
          *out << "IOSS-STK:  numDim =  " << numDim << endl;
        break;
      case stk::mesh::Node:
          {
            *out << "Mesh has Node Set ID: " << part->name() << endl;
            nsPartVec[part->name()]=part;
          }
        break;
      default: break ;
    }
  }

 *out << "IOSS-STK: number of node sets = " << nsPartVec.size() << endl;

 coordinates_field = metaData->get_field<VectorFieldType>(std::string("coordinates"));

  delete mesh_data;
  useElementAsTopRank = true;

  exoOutput = params->isType<string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<string>("Exodus Output File Name");
}

Albany::IossSTKMeshStruct::~IossSTKMeshStruct()
{ 
  delete metaData;
  delete bulkData;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::IossSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("ValidSTKIoss_DiscParams"));;
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");
  validPL->set<std::string>("Exodus Output File Name", "",
    "Request exodus output to given file name. Requires IOSS build");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");
  validPL->set<int>("Restart Index", 1, "Exodus time index to read for inital guess/condition.");

  return validPL;
}
#endif
