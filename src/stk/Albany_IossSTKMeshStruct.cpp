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
#include <Ionit_Initializer.h>

#include <stk_mesh/fem/FEMHelpers.hpp>

#include "Albany_Utils.hpp"

Albany::IossSTKMeshStruct::IossSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
		  const Teuchos::RCP<const Epetra_Comm>& comm) :
  GenericSTKMeshStruct(params),
  periodic(params->get("Periodic BC", false)),
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{

  params->validateParameters(*getValidDiscretizationParameters(),0);

  mesh_data = new stk::io::MeshData();

  Ioss::Init::Initializer io;

  usePamgen = (params->get("Method","Exodus") == "Pamgen");
  if (!usePamgen) {
    *out << "Albany_IOSS: Loading STKMesh from Exodus file  " 
         << params->get<string>("Exodus Input File Name") << endl;

    stk::io::create_input_mesh("exodusii",
                               params->get<string>("Exodus Input File Name"),
                               Albany::getMpiCommFromEpetraComm(*comm), 
                               *metaData, *mesh_data); 
    *out << "Albany_IOSS: Loading STKMesh from exodus file  " << endl;
  }
  else {
    *out << "Albany_IOSS: Loading STKMesh from Pamgen file  " 
         << params->get<string>("Pamgen Input File Name") << endl;

    stk::io::create_input_mesh("pamgen",
                               params->get<string>("Pamgen Input File Name"),
                               Albany::getMpiCommFromEpetraComm(*comm), 
                               *metaData, *mesh_data); 

  }

  numDim = metaData->spatial_dimension();
  *out << "IOSS-STK:  numDim =  " << numDim << endl;

  stk::io::put_io_part_attribute(metaData->universal_part());

  // Set element blocks and node sets
  const stk::mesh::PartVector & all_parts = metaData->get_parts();
  int eb=0;
  std::vector<std::string> nsNames;
  for (stk::mesh::PartVector::const_iterator i = all_parts.begin();
       i != all_parts.end(); ++i) {

    stk::mesh::Part * const part = *i ;

    if ( part->primary_entity_rank() == metaData->element_rank()) {
      *out << "IOSS-STK: Element part found " << endl;
      if (part->name()[0] != '{') partVec[eb++] = part;
    }
    else if ( part->primary_entity_rank() == metaData->node_rank()) {
       *out << "Mesh has Node Set ID: " << part->name() << endl;
      if (part->name()[0] != '{') {
         nsPartVec[part->name()]=part;
         nsNames.push_back(part->name());
      }
    }
  }

  int cub = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get("Workset Size",50);
  const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();

  // Get number of elements per element block using Ioss for use
  // in calculating an upper bound on the worksetSize.
  std::vector<int> el_blocks;
  stk::io::get_element_block_sizes(*mesh_data, el_blocks);
  TEST_FOR_EXCEPT(el_blocks.size() != partVec.size());

for (int i=0; i<el_blocks.size(); i++) cout << "BBB " << comm->MyPID() << "el block size " << el_blocks[i] << endl;
cout << "BBBBB " <<  comm->MyPID() << " Max is " << *std::max_element(el_blocks.begin(),el_blocks.end()) << endl;;

  int ebSizeMax =  *std::max_element(el_blocks.begin(),el_blocks.end());

  int worksetSize = this->computeWorksetSize(worksetSizeMax, ebSizeMax);
cout << "BBBCC " <<  comm->MyPID() << " Resized to  " << worksetSize << endl;

  // Construct MeshSpecsStruct
  this->meshSpecs = Teuchos::rcp(new Albany::MeshSpecsStruct(ctd, numDim, cub, nsNames, worksetSize));
}

void
Albany::IossSTKMeshStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize)
{
  this->SetupFieldData(comm, neq_, sis, worksetSize);

  *out << "IOSS-STK: number of node sets = " << nsPartVec.size() << endl;

  metaData->commit();

  stk::io::populate_bulk_data(*bulkData, *mesh_data);

  if (!usePamgen)  {
    // Restart index to read solution from exodus file.
    int index = params->get("Restart Index",-1); // Default to no restart
    if (index<1) *out << "Restart Index not set. Not reading solution from exodus (" 
           << index << ")"<< endl;
    else {
      *out << "Restart Index set, reading solution time : " << index << endl;
       stk::io::process_input_request(*mesh_data, *bulkData, 1.0*index);
    }
  }

  bulkData->modification_end();

  coordinates_field = metaData->get_field<VectorFieldType>(std::string("coordinates"));

  delete mesh_data;
  useElementAsTopRank = true;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::IossSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid IOSS_DiscParams");
  validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic a mesh");
  validPL->set<string>("Exodus Input File Name", "", "File Name For Exodus Mesh Input");
  validPL->set<string>("Pamgen Input File Name", "", "File Name For Pamgen Mesh Input");
  validPL->set<int>("Restart Index", 1, "Exodus time index to read for inital guess/condition.");

  return validPL;
}
#endif
