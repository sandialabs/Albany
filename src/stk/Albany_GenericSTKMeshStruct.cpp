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

  interleavedOrdering = params->get("Interleaved Ordering",true);

  // This is typical, can be resized for multiple material problems
  meshSpecs.resize(1);

  bulkData = NULL;
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

  if (bulkData ==  NULL)
  bulkData = new stk::mesh::BulkData(stk::mesh::fem::FEMMetaData::get_meta_data(*metaData),
                          Albany::getMpiCommFromEpetraComm(*comm), worksetSize );

  //Start STK stuff
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  solution_field = & metaData->declare_field< VectorFieldType >( "solution" );
  residual_field = & metaData->declare_field< VectorFieldType >( "residual" );

  stk::mesh::put_field( *coordinates_field , metaData->node_rank() , metaData->universal_part(), numDim );
  stk::mesh::put_field( *solution_field , metaData->node_rank() , metaData->universal_part(), neq );
  stk::mesh::put_field( *residual_field , metaData->node_rank() , metaData->universal_part() , neq );
  
#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*coordinates_field, Ioss::Field::MESH);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(*residual_field, Ioss::Field::TRANSIENT);
#endif

  // Code to parse the vector of StateStructs and create STK fields
  for (int i=0; i<sis->size(); i++) {
      Albany::StateStruct& st = *((*sis)[i]);
      std::vector<int>& dim = st.dim;
      if (dim.size() == 2 && st.entity=="QuadPoint") {
        qpscalar_states.push_back(& metaData->declare_field< QPScalarFieldType >( st.name) );
        stk::mesh::put_field( *qpscalar_states.back() , metaData->element_rank(),
                              metaData->universal_part(), dim[1]);
        cout << "NNNN qps field name " << qpscalar_states.back()->name() << endl;
#ifdef ALBANY_SEACAS
        if (st.output) stk::io::set_field_role(*qpscalar_states.back(), Ioss::Field::TRANSIENT);
#endif
      }
      else if (dim.size() == 3 && st.entity=="QuadPoint") {
        qpvector_states.push_back(& metaData->declare_field< QPVectorFieldType >( st.name) );
        // Multi-dim order is Fortran Ordering, so reversed here
        stk::mesh::put_field( *qpvector_states.back() , metaData->element_rank(),
                              metaData->universal_part(), dim[2], dim[1]);
        cout << "NNNN qpv field name " << qpvector_states.back()->name() << endl;
#ifdef ALBANY_SEACAS
        if (st.output) stk::io::set_field_role(*qpvector_states.back(), Ioss::Field::TRANSIENT);
#endif
      }
      else if (dim.size() == 4 && st.entity=="QuadPoint") {
        qptensor_states.push_back(& metaData->declare_field< QPTensorFieldType >( st.name) );
        // Multi-dim order is Fortran Ordering, so reversed here
        stk::mesh::put_field( *qptensor_states.back() , metaData->element_rank(),
                              metaData->universal_part(), dim[3], dim[2], dim[1]);
        cout << "NNNN qpt field name " << qptensor_states.back()->name() << endl;
#ifdef ALBANY_SEACAS
     if (st.output) stk::io::set_field_role(*qptensor_states.back(), Ioss::Field::TRANSIENT);
#endif
      }
     else TEST_FOR_EXCEPT(dim.size() < 2 || dim.size()>4 || st.entity!="QuadPoint");

  }
  
  // Exodus is only for 2D and 3D. Have 1D version as well
  if (numDim>1) {
    exoOutput = params->isType<string>("Exodus Output File Name");
    if (exoOutput)
      exoOutFile = params->get<string>("Exodus Output File Name");
    oneDOutput = false;
  }
  else if (numDim == 1) {
    oneDOutput = params->isType<string>("1D Output File Name");
    if (oneDOutput)
      oneDOutFile = params->get<string>("1D Output File Name");
    exoOutput = false;
  }
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

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >&
Albany::GenericSTKMeshStruct::getMeshSpecs()
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
  if (numDim==1)
    validPL->set<std::string>("1D Output File Name", "",
      "Request output of 1D solution and field info to this file.");
  validPL->set<std::string>("Method", "",
    "The discretization method, parsed in the Discretization Factory");
  validPL->set<int>("Cubature Degree", 3, "Integration order sent to Intrepid");
  validPL->set<int>("Workset Size", 50, "Upper bound on workset (bucket) size");
  validPL->set<bool>("Interleaved Ordering", true, "Flag for interleaved or blocked unknown ordering");

  return validPL;
}
