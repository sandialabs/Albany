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

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/fem/FieldDeclarations.hpp>
#include <stk_mesh/fem/TopologyHelpers.hpp>
#include <stk_mesh/fem/EntityRanks.hpp>


enum { field_data_chunk_size = 1001 };


Albany::FromCubitSTKMeshStruct::FromCubitSTKMeshStruct(
                  STKMeshData* stkMeshData,
                  const Teuchos::RCP<Teuchos::ParameterList>& params, 
                  const unsigned int neq_) :
  periodic(params->get("Periodic BC", false))
{
  params->validateParameters(*getValidDiscretizationParameters(),0);

  metaData = stkMeshData->get_meta_data();
  bulkData = stkMeshData->get_bulk_data();
  coordinates_field = stkMeshData->get_coords_field();
  solution_field = stkMeshData->get_solution_field();
  numDim = stkMeshData->get_high_rank();
  neq = neq_;

  // Construct nsPartVec from similar stkMeshData struct
  std::map<int, stk::mesh::Part*> nsList= stkMeshData->get_nodeset_list();
  std::map<int, stk::mesh::Part*>::iterator ns = nsList.begin();
  while ( ns != nsList.end() ) {
    // Name chosen to be same as Ioss default "nodelist_" + <int>
    std::stringstream ss; ss << "nodelist_" << ns->first;
    nsPartVec[ss.str()] = ns->second;
    ns++;
  }


  cubatureDegree = params->get("Cubature Degree", 3);

  partVec[0] = & metaData->universal_part();

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
  validPL->set<std::string>("Exodus Output File Name", "",
    "Request exodus output to given file name. Requires IOSS build");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");

  return validPL;
}

#endif
