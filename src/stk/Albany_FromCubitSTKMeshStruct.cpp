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

#ifdef ALBANY_CUTR

#include "Albany_FromCubitSTKMeshStruct.hpp"

// STKMesh interface loaded by Cubit MeshMover
#include "STKMeshData.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/fem/FieldDeclarations.hpp>
#include <stk_mesh/fem/FEMHelpers.hpp>
#include <stk_mesh/fem/EntityRanks.hpp>
#include "Albany_Utils.hpp"

enum { field_data_chunk_size = 1001 };


Albany::FromCubitSTKMeshStruct::FromCubitSTKMeshStruct(
                  const Teuchos::RCP<CUTR::CubitMeshMover>& meshMover, 
                  const Teuchos::RCP<Teuchos::ParameterList>& params, 
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
) :
  periodic(params->get("Periodic BC", false))
{
  params->validateParameters(*getValidDiscretizationParameters(),0);
  neq=neq_;
  nstates=sis->nstates;

  // Get singleton to STK info as loaded by Cubit MeshMover
  STKMeshData* stkMeshData = STKMeshData::instance();

  metaData = stkMeshData->get_meta_data();
  coordinates_field = stkMeshData->get_coords_field();

  solution_field = & metaData->declare_field< VectorFieldType >( "solution" );
  residual_field = & metaData->declare_field< VectorFieldType >( "residual" );
  state_field = & metaData->declare_field< VectorFieldType >( "state" );
  stk::mesh::put_field( *solution_field , metaData->node_rank() , metaData->universal_part(), neq );
  stk::mesh::put_field( *residual_field , metaData->node_rank() , metaData->universal_part() , neq );
  if (nstates>0) stk::mesh::put_field( *state_field , metaData->element_rank() , metaData->universal_part(), nstates );

  // Construct nsPartVec from similar stkMeshData struct
  std::map<int, stk::mesh::Part*> nsList= stkMeshData->get_nodeset_list();
  std::map<int, stk::mesh::Part*>::iterator ns = nsList.begin();
  while ( ns != nsList.end() ) {
    // Name chosen to be same as Ioss default "nodelist_" + <int>
    std::stringstream ss; ss << "nodelist_" << ns->first;
    nsPartVec[ss.str()] = ns->second;
#ifdef ALBANY_IOSS
   stk::io::put_io_part_attribute(*ns->second);
#endif
    ns++;
  }

  numDim = stkMeshData->get_num_dim();
  cout << "numDim form cubit  " << numDim << endl;

  if (numDim==2) partVec[0] = stkMeshData->surface_part(0);
  else           partVec[0] = stkMeshData->volume_part(0);

#ifdef ALBANY_IOSS
/*
  // set all top rank parts as IO parts
  int id=0;
  stk::mesh::Part* eb;
  do {
    if (numDim==2) eb = stkMeshData->surface_part(id++);
    else           eb = stkMeshData->volume_part(id++);
    stk::io::put_io_part_attribute(*eb);
  } while (eb!=NULL);
*/

  stk::io::set_field_role(*coordinates_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(*residual_field, Ioss::Field::TRANSIENT);
  if (nstates>0) stk::io::set_field_role(*state_field, Ioss::Field::TRANSIENT);
#endif

  // This calls metaData->commit()
  stkMeshData->declare_geometry_complete();

  // Load and get bulk data from Cubit
  stkMeshData->begin_declare_entities();
  meshMover->construct_bulkdata();
  stkMeshData->end_declare_entities();
  bulkData = stkMeshData->get_bulk_data();

  meshMover->set_morph_method(params->get("Morph Method", 0));

  cubatureDegree = params->get("Cubature Degree", 3);

  cout << "FromCubitSTKMeshStruct: numDim = " << numDim
       << "  nodesets = " << nsPartVec.size() << endl;
  useElementAsTopRank = false;

  exoOutput = params->isType<string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<string>("Exodus Output File Name");
}

Albany::FromCubitSTKMeshStruct::~FromCubitSTKMeshStruct()
{
}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::FromCubitSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("ValidCubit_DiscParams"));;
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<int>("Morph Method", 0, "Integer flag so select CUTR MeshMover Morph Method");
  validPL->set<std::string>("Exodus Output File Name", "",
    "Request exodus output to given file name. Requires IOSS build");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");

  return validPL;
}

#endif
