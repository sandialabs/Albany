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

#include "Albany_GenericSTKMeshStruct.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <stk_mesh/fem/FEMHelpers.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif
#include "Albany_Utils.hpp"

Albany::GenericSTKMeshStruct::GenericSTKMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    int numDim_)
    : params(params_)
{
  metaData = new stk::mesh::fem::FEMMetaData();
  
  // numDim = -1 is default flag value to postpone initialization
  if (numDim_>0) {
    this->numDim = numDim_;
    metaData->FEM_initialize(numDim_);
  }

  bulkData = NULL;

  cubatureDegree = params->get("Cubature Degree", 3);
}

void Albany::GenericSTKMeshStruct::SetupFieldData(
		  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const int worksetSize) 
{
  TEST_FOR_EXCEPTION(!metaData->is_FEM_initialized(),
       std::logic_error,
       "LogicError: metaData->FEM_initialize(numDim) not yet called" << std::endl);

  neq = neq_;
  nstates = sis->nstates;

  if (bulkData ==  NULL)
  bulkData = new stk::mesh::BulkData(stk::mesh::fem::FEMMetaData::get_meta_data(*metaData),
                          Albany::getMpiCommFromEpetraComm(*comm), worksetSize );

  //Start STK stuff
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  solution_field = & metaData->declare_field< VectorFieldType >( "solution" );
  residual_field = & metaData->declare_field< VectorFieldType >( "residual" );
  state_field = & metaData->declare_field< VectorFieldType >( "state" );

  stk::mesh::put_field( *coordinates_field , metaData->node_rank() , metaData->universal_part(), numDim );
  stk::mesh::put_field( *solution_field , metaData->node_rank() , metaData->universal_part(), neq );
  stk::mesh::put_field( *residual_field , metaData->node_rank() , metaData->universal_part() , neq );
  if (nstates>0) stk::mesh::put_field( *state_field , metaData->element_rank() , metaData->universal_part(), nstates );
  
#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*coordinates_field, Ioss::Field::MESH);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(*residual_field, Ioss::Field::TRANSIENT);
  if (nstates>0) stk::io::set_field_role(*state_field, Ioss::Field::TRANSIENT);
#endif

  exoOutput = params->isType<string>("Exodus Output File Name");
  if (exoOutput)
    exoOutFile = params->get<string>("Exodus Output File Name");
}

void Albany::GenericSTKMeshStruct::DeclareParts(std::vector<std::string> nsNames)
{
  // HandCoded meshes have 1 element block
  partVec[0] = &  metaData->declare_part( "Block_1", metaData->element_rank() );
#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(*partVec[0]);
#endif

  // NodeSets
  for (unsigned int i=0; i<nsNames.size(); i++) {
    std::string nsn = nsNames[i];
    nsPartVec[nsn] = & metaData->declare_part(nsn, metaData->node_rank() );
#ifdef ALBANY_SEACAS
    stk::io::put_io_part_attribute(*nsPartVec[nsn]);
#endif
  }
}


Albany::GenericSTKMeshStruct::~GenericSTKMeshStruct()
{
  delete metaData;
  delete bulkData;
}

const Teuchos::RCP<Albany::MeshSpecsStruct>&
Albany::GenericSTKMeshStruct::getMeshSpecs() const
{
  TEST_FOR_EXCEPTION(meshSpecs==Teuchos::null,
       std::logic_error,
       "meshSpecs accessed, but it has not been constructed" << std::endl);
  return meshSpecs;
}

int Albany::GenericSTKMeshStruct::computeWorksetSize(const int worksetSizeMax,
                                                     const int ebSizeMax) const
{
  // Resize workset size down to maximum number in an element block
  if (worksetSizeMax > ebSizeMax || worksetSizeMax < 1) return ebSizeMax;
  else {
     // compute numWorksets, and shrink workset size to minimize padding
     const int numWorksets = 1 + (ebSizeMax-1) / worksetSizeMax;
     return (1 + (ebSizeMax-1) /  numWorksets);
  }
}


Teuchos::RCP<Teuchos::ParameterList>
Albany::GenericSTKMeshStruct::getValidGenericSTKParameters(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = rcp(new Teuchos::ParameterList(listname));;
  validPL->set<string>("Cell Topology", "Quad" , "Quad or Tri Cell Topology");
  validPL->set<std::string>("Exodus Output File Name", "",
    "Request exodus output to given file name. Requires SEACAS build");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");
  validPL->set<int>("Workset Size", 50, "Upper bound on workset (bucket) size");

  return validPL;
}
