//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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

  // Get singleton to STK info as loaded by Cubit MeshMover
  STKMeshData* stkMeshData = STKMeshData::instance();

  metaData = stkMeshData->get_meta_data();
  coordinates_field = stkMeshData->get_coords_field();

  solution_field = & metaData->declare_field< VectorFieldType >( "solution" );
  residual_field = & metaData->declare_field< VectorFieldType >( "residual" );
  stk::mesh::put_field( *solution_field , metaData->node_rank() , metaData->universal_part(), neq );
  stk::mesh::put_field( *residual_field , metaData->node_rank() , metaData->universal_part() , neq );

  // Construct nsPartVec from similar stkMeshData struct
  std::map<int, stk::mesh::Part*> nsList= stkMeshData->get_nodeset_list();
  std::map<int, stk::mesh::Part*>::iterator ns = nsList.begin();
  while ( ns != nsList.end() ) {
    // Name chosen to be same as Ioss default "nodelist_" + <int>
    std::stringstream ss; ss << "nodelist_" << ns->first;
    nsPartVec[ss.str()] = ns->second;
#ifdef ALBANY_SEACAS
   stk::io::put_io_part_attribute(*ns->second);
#endif
    ns++;
  }

  numDim = stkMeshData->get_num_dim();

  if (numDim==2) partVec[0] = stkMeshData->surface_part(0);
  else           partVec[0] = stkMeshData->volume_part(0);

#ifdef ALBANY_SEACAS
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
  cdfOutput = params->isType<string>("NetCDF Output File Name");
  if (cdfOutput) 
    cdfOutFile = params->get<string>("NetCDF Output File Name");
  nLat       =  params->get("NetCDF Output Number of Latitudes",100);
  nLon       =  params->get("NetCDF Output Number of Longitudes",100);
  
  //get the type of transformation of STK mesh (for FELIX problems)
  transformType = params->get("Transform Type", "None"); //get the type of transformation of STK mesh (for FELIX problems)
  felixAlpha = params->get("FELIX alpha", 0.0); 
  felixL = params->get("FELIX L", 1.0); 
  
  //boolean specifying if ascii mesh has contiguous IDs; only used for ascii meshes on 1 processor
  contigIDs = params->get("Contiguous IDs", true);
  
  //Does user want to write coordinates to matrix market file (e.g., for ML analysis)? 
  writeCoordsToMMFile = params->get("Write Coordinates to MatrixMarket", false); 
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
    "Request exodus output to given file name. Requires SEACAS build");
  validPL->set<std::string>("NetCDF Output File Name", "",
    "Request netcdf output to given file name. Requires SEACAS build");
  validPL->set<int>("NetCDF Output Number of Latitudes", 1, 
    "Number of samples in Latitude direction for NetCDF output. Default is 100.");
  validPL->set<int>("NetCDF Output Number of Longitudes", 1, 
    "Number of samples in Longitude direction for NetCDF output. Default is 100.");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");

  return validPL;
}

#endif
