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

  metaData = new stk::mesh::MetaData(stk::mesh::fem_entity_rank_names() );
  bulkData = new stk::mesh::BulkData(*metaData , MPI_COMM_WORLD , field_data_chunk_size );
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  solution_field = & metaData->declare_field< VectorFieldType >( "solution" );
  stk::mesh::put_field( *solution_field , stk::mesh::Node , metaData->universal_part() , neq );
  stk::io::set_field_role(*coordinates_field, Ioss::Field::ATTRIBUTE);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);

  stk::io::util::MeshData* mesh_data = new stk::io::util::MeshData();

  Ioss::Init::Initializer io;

  bool usePamgen = (params->get("Method","Exodus") == "Pamgen");
  if (!usePamgen) {
    *out << "Albany_IOSS: Loading STKMesh from Exodus file  " 
         << params->get<string>("Exodus Input File Name") << endl;

    stk::io::util::create_input_mesh("exodusii",
                               params->get<string>("Exodus Input File Name"),
                               "", MPI_COMM_WORLD, 
                               *metaData, *mesh_data, false); 
    *out << "Albany_IOSS: Loading STKMesh from exodus file  " << endl;
  }
  else {
    *out << "Albany_IOSS: Loading STKMesh from Pamgen file  " 
         << params->get<string>("Pamgen Input File Name") << endl;

    stk::io::util::create_input_mesh("pamgen",
                               params->get<string>("Pamgen Input File Name"),
                               "", MPI_COMM_WORLD, 
                               *metaData, *mesh_data, false); 

  }


  stk::io::put_io_part_attribute(metaData->universal_part());

  // Set node sets
  const stk::mesh::PartVector & all_parts = metaData->get_parts();
  int eb=0;
  for (stk::mesh::PartVector::const_iterator i = all_parts.begin();
       i != all_parts.end(); ++i) {

    stk::mesh::Part * const part = *i ;

    switch( part->primary_entity_rank() ) {
      case stk::mesh::Element:{
          *out << "IOSS-STK: Element part found " << endl;
          partVec[eb++] = part;
          // Since Cubit likes to define numDim=3 always, use vertex
          // count on top element block to figure out quad(tri) vs hex.
          //   Needs to be fixed for Tets ro Shells
          int numVerts = stk::mesh::get_cell_topology(*part)->vertex_count;
          if (numVerts==4 || numVerts==3) numDim=2;
          else if (numVerts==8) numDim=3;
          else TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                 std::endl << "Error!  IossSTKMeshStruct:  " <<
                 "Invalid vertex count from exodus mesh: " << numVerts << std::endl);
          *out << "IOSS-STK:  numDim =  " << numDim << endl; }
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

  metaData->commit();

  if (!usePamgen)  {
    // Restart index to read solution from exodus file.
    int index = params->get("Restart Index",-1); // Default to no restart
    if (index<1) *out << "Restart Index not set. Not reading solution from exodus (" 
           << index << ")"<< endl;
    else *out << "Restart Index set, reading solution time step: " << index << endl;

    stk::io::util::populate_bulk_data(*bulkData, *mesh_data, "exodusii", index);
  }
  else  stk::io::util::populate_bulk_data(*bulkData, *mesh_data, "pamgen", 0);

  bulkData->modification_end();

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
  validPL->set<string>("Pamgen Input File Name", "", "File Name For Pamgen Mesh Input");
  validPL->set<std::string>("Exodus Output File Name", "",
    "Request exodus output to given file name. Requires IOSS build");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");
  validPL->set<int>("Restart Index", 1, "Exodus time index to read for inital guess/condition.");

  return validPL;
}
#endif
