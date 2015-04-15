//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include <iostream>

#include "Albany_OrdinarySTKFieldContainer.hpp"

#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetBuckets.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

template<bool Interleaved>
Albany::OrdinarySTKFieldContainer<Interleaved>::OrdinarySTKFieldContainer(
  const Teuchos::RCP<Teuchos::ParameterList>& params_,
  const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
  const int neq_,
  const AbstractFieldContainer::FieldContainerRequirements& req,
  const int numDim_,
  const Teuchos::RCP<Albany::StateInfoStruct>& sis)
  : GenericSTKFieldContainer<Interleaved>(params_, metaData_, neq_, numDim_),
      buildSphereVolume(false) {

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType SFT;

  //Start STK stuff
  this->coordinates_field = & metaData_->declare_field< VFT >(stk::topology::NODE_RANK, "coordinates");
  solution_field = & metaData_->declare_field< VFT >(stk::topology::NODE_RANK,
                                                     params_->get<std::string>("Exodus Solution Name", "solution"));

#if defined(ALBANY_LCM)
  residual_field = & metaData_->declare_field< VFT >(stk::topology::NODE_RANK,
                                                     params_->get<std::string>("Exodus Residual Name", "residual"));
#endif

  stk::mesh::put_field(*this->coordinates_field , metaData_->universal_part(), numDim_);
  stk::mesh::put_field(*solution_field , metaData_->universal_part(), neq_);

#if defined(ALBANY_LCM)
  stk::mesh::put_field(*residual_field , metaData_->universal_part() , neq_);
#endif

#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*this->coordinates_field, Ioss::Field::MESH);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
#if defined(ALBANY_LCM)
  stk::io::set_field_role(*residual_field, Ioss::Field::TRANSIENT);
#endif

#endif

#if defined(ALBANY_LCM)
  // sphere volume is a mesh attribute read from a genesis mesh file containing sphere element (used for peridynamics)
  // for whatever reason, its type is stk::mesh::Field<double, stk::mesh::Cartesian3d>
  // the read won't work if you try to read it as a SFT
  bool hasSphereVolumeFieldContainerRequirement = (std::find(req.begin(), req.end(), "Sphere Volume") != req.end());
  if(hasSphereVolumeFieldContainerRequirement){
    this->sphereVolume_field = metaData_->template get_field< stk::mesh::Field<double, stk::mesh::Cartesian3d> >(stk::topology::ELEMENT_RANK, "volume");
    if(this->sphereVolume_field != 0){
      buildSphereVolume = true;
      stk::io::set_field_role(*this->sphereVolume_field, Ioss::Field::ATTRIBUTE);
    }
  }
#endif

  // If the problem requests that the initial guess at the solution equals the input node coordinates,
  // set that here
  /*
    if(std::find(req.begin(), req.end(), "Initial Guess Coords") != req.end()){
       this->copySTKField(this->coordinates_field, solution_field);
    }
  */

  this->buildStateStructs(sis);

  initializeSTKAdaptation();
}

template<bool Interleaved>
Albany::OrdinarySTKFieldContainer<Interleaved>::~OrdinarySTKFieldContainer() {
}

template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::initializeSTKAdaptation() {

  typedef typename AbstractSTKFieldContainer::IntScalarFieldType ISFT;

  this->proc_rank_field =
    & this->metaData->template declare_field< ISFT >(stk::topology::ELEMENT_RANK, "proc_rank");

  this->refine_field =
    & this->metaData->template declare_field< ISFT >(stk::topology::ELEMENT_RANK, "refine_field");

  // Processor rank field, a scalar
  stk::mesh::put_field(
      *this->proc_rank_field,
      this->metaData->universal_part());

  stk::mesh::put_field(
      *this->refine_field,
      this->metaData->universal_part());

#if defined(ALBANY_LCM)
  // Fracture state used for adaptive insertion.
  // It exists for all entities except cells (elements).
  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::ELEMENT_RANK; ++rank) {
    this->fracture_state[rank] = & this->metaData->template declare_field< ISFT >(rank, "fracture_state");

    stk::mesh::put_field(
        *this->fracture_state[rank],
        this->metaData->universal_part());

  }
#endif // ALBANY_LCM

#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*this->proc_rank_field, Ioss::Field::MESH);
  stk::io::set_field_role(*this->refine_field, Ioss::Field::MESH);
#if defined(ALBANY_LCM)
  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank < stk::topology::ELEMENT_RANK; ++rank) {
    stk::io::set_field_role(*this->fracture_state[rank], Ioss::Field::MESH);
  }
#endif // ALBANY_LCM
#endif

}

#ifdef ALBANY_EPETRA
template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::fillSolnVector(Epetra_Vector& soln,
    stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) {

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::BulkData& mesh = solution_field->get_mesh();
  stk::mesh::BucketVector const& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, sel);
  this->numNodes = node_map->NumMyElements(); // Needed for the getDOF function to work correctly
  // This is either numOwnedNodes or numOverlapNodes, depending on
  // which map is passed in

  for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {

    const stk::mesh::Bucket& bucket = **it;

    this->fillVectorHelper(soln, solution_field, node_map, bucket, 0);

  }

}
#endif

//Tpetra version of above
template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::fillSolnVectorT(Tpetra_Vector &solnT,
       stk::mesh::Selector &sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT){

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::BulkData& mesh = solution_field->get_mesh();
  stk::mesh::BucketVector const& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, sel);
  this->numNodes = node_mapT->getNodeNumElements(); // Needed for the getDOF function to work correctly
                                        // This is either numOwnedNodes or numOverlapNodes, depending on
                                        // which map is passed in

   for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {

    const stk::mesh::Bucket& bucket = **it;

    this->fillVectorHelperT(solnT, solution_field, node_mapT, bucket, 0);

  }
}

#ifdef ALBANY_EPETRA
template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::fillVector(Epetra_Vector& field_vector, const std::string&  field_name,
    stk::mesh::Selector& field_selection, const Teuchos::RCP<Epetra_Map>& field_node_map, const NodalDOFManager& nodalDofManager) {

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::BulkData& mesh = solution_field->get_mesh();

  stk::mesh::BucketVector const& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, field_selection);

  if(nodalDofManager.numComponents() > 1) {
    AbstractSTKFieldContainer::VectorFieldType* field  = mesh.mesh_meta_data().get_field<AbstractSTKFieldContainer::VectorFieldType>(stk::topology::NODE_RANK, field_name);
    for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {
      const stk::mesh::Bucket& bucket = **it;
      this->fillVectorHelper(field_vector, field, field_node_map, bucket, nodalDofManager);
    }
  }
  else {
    AbstractSTKFieldContainer::ScalarFieldType* field  = mesh.mesh_meta_data().get_field<AbstractSTKFieldContainer::ScalarFieldType>(stk::topology::NODE_RANK, field_name);
    for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {
      const stk::mesh::Bucket& bucket = **it;
      this->fillVectorHelper(field_vector, field, field_node_map, bucket, nodalDofManager);
    }
  }
}

template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::saveVector(const Epetra_Vector& field_vector, const std::string&  field_name,
    stk::mesh::Selector& field_selection, const Teuchos::RCP<Epetra_Map>& field_node_map, const NodalDOFManager& nodalDofManager) {

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::BulkData& mesh = solution_field->get_mesh();
  stk::mesh::BucketVector const& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, field_selection);

  if(nodalDofManager.numComponents() > 1) {
    AbstractSTKFieldContainer::VectorFieldType* field  = mesh.mesh_meta_data().get_field<AbstractSTKFieldContainer::VectorFieldType>(stk::topology::NODE_RANK, field_name);
    for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {
      const stk::mesh::Bucket& bucket = **it;
      this->saveVectorHelper(field_vector, field, field_node_map, bucket, nodalDofManager);
    }
  }
  else {
    AbstractSTKFieldContainer::ScalarFieldType* field  = mesh.mesh_meta_data().get_field<AbstractSTKFieldContainer::ScalarFieldType>(stk::topology::NODE_RANK, field_name);
    for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {
      const stk::mesh::Bucket& bucket = **it;
      this->saveVectorHelper(field_vector, field, field_node_map, bucket, nodalDofManager);
    }
  }
}

template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::saveSolnVector(const Epetra_Vector& soln,
    stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) {

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::BulkData& mesh = solution_field->get_mesh();
  stk::mesh::BucketVector const& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, sel);
  this->numNodes = node_map->NumMyElements(); // Needed for the getDOF function to work correctly
  // This is either numOwnedNodes or numOverlapNodes, depending on
  // which map is passed in

  for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {

    const stk::mesh::Bucket& bucket = **it;

    this->saveVectorHelper(soln, solution_field, node_map, bucket, 0);

  }

}
#endif

//Tpetra version of above
template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::saveSolnVectorT(const Tpetra_Vector& solnT,
    stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT) {


  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::BulkData& mesh = solution_field->get_mesh();
  stk::mesh::BucketVector const& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, sel);
  this->numNodes = node_mapT->getNodeNumElements(); // Needed for the getDOF function to work correctly
  // This is either numOwnedNodes or numOverlapNodes, depending on
  // which map is passed in

   for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {

    const stk::mesh::Bucket& bucket = **it;

    this->saveVectorHelperT(solnT, solution_field, node_mapT, bucket, 0);

  }

}

#ifdef ALBANY_EPETRA
template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::saveResVector(const Epetra_Vector& res,
    stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) {

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::BulkData& mesh = solution_field->get_mesh();
  stk::mesh::BucketVector const& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, sel);
  this->numNodes = node_map->NumMyElements(); // Needed for the getDOF function to work correctly
  // This is either numOwnedNodes or numOverlapNodes, depending on
  // which map is passed in

  for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {

    const stk::mesh::Bucket& bucket = **it;

    this->saveVectorHelper(res, residual_field, node_map, bucket, 0);

  }

}
#endif

template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::saveResVectorT(const Tpetra_Vector& res,
    stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_map) {

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::BulkData& mesh = solution_field->get_mesh();
  stk::mesh::BucketVector const& all_elements = mesh.get_buckets(stk::topology::NODE_RANK, sel);
  this->numNodes = node_map->getNodeNumElements(); // Needed for the getDOF function to work correctly
  // This is either numOwnedNodes or numOverlapNodes, depending on
  // which map is passed in

  for(stk::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {

    const stk::mesh::Bucket& bucket = **it;

    this->saveVectorHelperT(res, residual_field, node_map, bucket, 0);

  }

}

template<bool Interleaved>
void Albany::OrdinarySTKFieldContainer<Interleaved>::transferSolutionToCoords() {

  this->copySTKField(solution_field, this->coordinates_field);

}

