//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#include "Albany_MultiSTKFieldContainer.hpp"

#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetBuckets.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

template<bool Interleaved>
Albany::MultiSTKFieldContainer<Interleaved>::MultiSTKFieldContainer(
  const Teuchos::RCP<Teuchos::ParameterList>& params_,
  stk_classic::mesh::fem::FEMMetaData* metaData_,
  stk_classic::mesh::BulkData* bulkData_,
  const int neq_,
  const AbstractFieldContainer::FieldContainerRequirements& req,
  const int numDim_,
  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
  const Teuchos::Array<std::string>& solution_vector,
  const Teuchos::Array<std::string>& residual_vector)
  : GenericSTKFieldContainer<Interleaved>(params_, metaData_, bulkData_, neq_, numDim_),
    haveResidual(false), buildSphereVolume(false) {

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType SFT;

#ifdef ALBANY_LCM
  buildSphereVolume = (std::find(req.begin(), req.end(), "Sphere Volume") != req.end());
#endif

  // Check the input

  if(solution_vector.size() == 0) { // Do the default solution vector

    std::string name = params_->get<std::string>("Exodus Solution Name", "solution");
    VFT* solution = & metaData_->declare_field< VFT >(name);
    stk_classic::mesh::put_field(*solution, metaData_->node_rank() , metaData_->universal_part(), neq_);
#ifdef ALBANY_SEACAS
    stk_classic::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
#endif

    sol_vector_name.push_back(name);
    sol_index.push_back(this->neq);

  }

  else if(solution_vector.size() == 1) { // User is just renaming the entire solution vector

    VFT* solution = & metaData_->declare_field< VFT >(solution_vector[0]);
    stk_classic::mesh::put_field(*solution, metaData_->node_rank() , metaData_->universal_part(), neq_);
#ifdef ALBANY_SEACAS
    stk_classic::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
#endif

    sol_vector_name.push_back(solution_vector[0]);
    sol_index.push_back(neq_);

  }

  else { // user is breaking up the solution into multiple fields

    // make sure the number of entries is even

    TEUCHOS_TEST_FOR_EXCEPTION((solution_vector.size() % 2), std::logic_error,
                               "Error in input file: specification of solution vector layout is incorrect." << std::endl);

    int len, accum = 0;

    for(int i = 0; i < solution_vector.size(); i += 2) {

      if(solution_vector[i + 1] == "V") {

        len = numDim_; // vector
        accum += len;
        VFT* solution = & metaData_->declare_field< VFT >(solution_vector[i]);
        stk_classic::mesh::put_field(*solution, metaData_->node_rank() , metaData_->universal_part(), len);
#ifdef ALBANY_SEACAS
        stk_classic::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
#endif
        sol_vector_name.push_back(solution_vector[i]);
        sol_index.push_back(len);

      }

      else if(solution_vector[i + 1] == "S") {

        len = 1; // scalar
        accum += len;
        SFT* solution = & metaData_->declare_field< SFT >(solution_vector[i]);
        stk_classic::mesh::put_field(*solution, metaData_->node_rank() , metaData_->universal_part());
#ifdef ALBANY_SEACAS
        stk_classic::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
#endif
        sol_vector_name.push_back(solution_vector[i]);
        sol_index.push_back(len);

      }

      else

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                   "Error in input file: specification of solution vector layout is incorrect." << std::endl);

    }

    TEUCHOS_TEST_FOR_EXCEPTION(accum != neq_, std::logic_error,
                               "Error in input file: specification of solution vector layout is incorrect." << std::endl);

  }

#ifdef ALBANY_LCM
  // do the residual next

  if(residual_vector.size() == 0) { // Do the default residual vector

    std::string name = params_->get<std::string>("Exodus Residual Name", "residual");
    VFT* residual = & metaData_->declare_field< VFT >(name);
    stk_classic::mesh::put_field(*residual, metaData_->node_rank() , metaData_->universal_part(), neq_);
#ifdef ALBANY_SEACAS
    stk_classic::io::set_field_role(*residual, Ioss::Field::TRANSIENT);
#endif

    res_vector_name.push_back(name);
    res_index.push_back(neq_);

  }

  else if(residual_vector.size() == 1) { // User is just renaming the entire residual vector

    VFT* residual = & metaData_->declare_field< VFT >(residual_vector[0]);
    stk_classic::mesh::put_field(*residual, metaData_->node_rank() , metaData_->universal_part(), neq_);
#ifdef ALBANY_SEACAS
    stk_classic::io::set_field_role(*residual, Ioss::Field::TRANSIENT);
#endif

    res_vector_name.push_back(residual_vector[0]);
    res_index.push_back(neq_);

  }

  else { // user is breaking up the residual into multiple fields

    // make sure the number of entries is even

    TEUCHOS_TEST_FOR_EXCEPTION((residual_vector.size() % 2), std::logic_error,
                               "Error in input file: specification of residual vector layout is incorrect." << std::endl);

    int len, accum = 0;

    for(int i = 0; i < residual_vector.size(); i += 2) {

      if(residual_vector[i + 1] == "V") {

        len = numDim_; // vector
        accum += len;
        VFT* residual = & metaData_->declare_field< VFT >(residual_vector[i]);
        stk_classic::mesh::put_field(*residual, metaData_->node_rank() , metaData_->universal_part(), len);
#ifdef ALBANY_SEACAS
        stk_classic::io::set_field_role(*residual, Ioss::Field::TRANSIENT);
#endif
        res_vector_name.push_back(residual_vector[i]);
        res_index.push_back(len);

      }

      else if(residual_vector[i + 1] == "S") {

        len = 1; // scalar
        accum += len;
        SFT* residual = & metaData_->declare_field< SFT >(residual_vector[i]);
        stk_classic::mesh::put_field(*residual, metaData_->node_rank() , metaData_->universal_part());
#ifdef ALBANY_SEACAS
        stk_classic::io::set_field_role(*residual, Ioss::Field::TRANSIENT);
#endif
        res_vector_name.push_back(residual_vector[i]);
        res_index.push_back(len);

      }

      else

        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                   "Error in input file: specification of residual vector layout is incorrect." << std::endl);

    }

    TEUCHOS_TEST_FOR_EXCEPTION(accum != neq_, std::logic_error,
                               "Error in input file: specification of residual vector layout is incorrect." << std::endl);

  }

  haveResidual = true;

#endif

  //Do the coordinates
  this->coordinates_field = & metaData_->declare_field< VFT >("coordinates");
  stk_classic::mesh::put_field(*this->coordinates_field , metaData_->node_rank() , metaData_->universal_part(), numDim_);
#ifdef ALBANY_SEACAS
  stk_classic::io::set_field_role(*this->coordinates_field, Ioss::Field::MESH);
#endif

#ifdef ALBANY_LCM
  // sphere volume is a mesh attribute read from a genesis mesh file containing sphere element (used for peridynamics)
  if(buildSphereVolume){
    this->sphereVolume_field = metaData_->get_field< stk_classic::mesh::Field<double> >("volume");
    if(this->sphereVolume_field)
      stk_classic::io::set_field_role(*this->sphereVolume_field, Ioss::Field::ATTRIBUTE);
  }
#endif

  this->buildStateStructs(sis);

  initializeSTKAdaptation();

}

template<bool Interleaved>
Albany::MultiSTKFieldContainer<Interleaved>::~MultiSTKFieldContainer() {
}

template<bool Interleaved>
void Albany::MultiSTKFieldContainer<Interleaved>::initializeSTKAdaptation() {

  typedef typename AbstractSTKFieldContainer::IntScalarFieldType ISFT;

  this->proc_rank_field =
      & this->metaData->template declare_field< ISFT >("proc_rank");

  this->refine_field =
      & this->metaData->template declare_field< ISFT >("refine_field");

  // Processor rank field, a scalar
  stk_classic::mesh::put_field(
      *this->proc_rank_field,
      this->metaData->element_rank(),
      this->metaData->universal_part());

  stk_classic::mesh::put_field(
      *this->refine_field,
      this->metaData->element_rank(),
      this->metaData->universal_part());

#ifdef ALBANY_LCM
  // Fracture state used for adaptive insertion.
  // It exists for all entities except cells (elements).
  this->fracture_state =
      & this->metaData->template declare_field< ISFT >("fracture_state");

  stk_classic::mesh::EntityRank const
  cell_rank = this->metaData->element_rank();

  for (stk_classic::mesh::EntityRank rank = 0; rank < cell_rank; ++rank) {
    stk_classic::mesh::put_field(
        *this->fracture_state,
        rank,
        this->metaData->universal_part());

  }
#endif // ALBANY_LCM


#ifdef ALBANY_SEACAS
  stk_classic::io::set_field_role(*this->proc_rank_field, Ioss::Field::MESH);
  stk_classic::io::set_field_role(*this->refine_field, Ioss::Field::MESH);
#ifdef ALBANY_LCM
  stk_classic::io::set_field_role(*this->fracture_state, Ioss::Field::MESH);
#endif // ALBANY_LCM
#endif

}

template<bool Interleaved>
void Albany::MultiSTKFieldContainer<Interleaved>::fillSolnVector(Epetra_Vector& soln,
    stk_classic::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) {

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType SFT;

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk_classic::mesh::BucketVector all_elements;
  stk_classic::mesh::get_buckets(sel, this->bulkData->buckets(this->metaData->node_rank()), all_elements);
  this->numNodes = node_map->NumMyElements(); // Needed for the getDOF function to work correctly
  // This is either numOwnedNodes or numOverlapNodes, depending on
  // which map is passed in

  for(stk_classic::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {

    const stk_classic::mesh::Bucket& bucket = **it;

    int offset = 0;

    for(int k = 0; k < sol_index.size(); k++) {

      if(sol_index[k] == 1) { // Scalar

        SFT* field = this->metaData->template get_field<SFT>(sol_vector_name[k]);
        this->fillVectorHelper(soln, field, node_map, bucket, offset);

      }

      else {

        VFT* field = this->metaData->template get_field<VFT>(sol_vector_name[k]);
        this->fillVectorHelper(soln, field, node_map, bucket, offset);

      }

      offset += sol_index[k];

    }

  }
}

template<bool Interleaved>
void Albany::MultiSTKFieldContainer<Interleaved>::saveSolnVector(const Epetra_Vector& soln,
    stk_classic::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) {

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType SFT;

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk_classic::mesh::BucketVector all_elements;
  stk_classic::mesh::get_buckets(sel, this->bulkData->buckets(this->metaData->node_rank()), all_elements);
  this->numNodes = node_map->NumMyElements(); // Needed for the getDOF function to work correctly
  // This is either numOwnedNodes or numOverlapNodes, depending on
  // which map is passed in

  for(stk_classic::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {

    const stk_classic::mesh::Bucket& bucket = **it;

    int offset = 0;

    for(int k = 0; k < sol_index.size(); k++) {

      if(sol_index[k] == 1) { // Scalar

        SFT* field = this->metaData->template get_field<SFT>(sol_vector_name[k]);
        this->saveVectorHelper(soln, field, node_map, bucket, offset);

      }

      else {

        VFT* field = this->metaData->template get_field<VFT>(sol_vector_name[k]);
        this->saveVectorHelper(soln, field, node_map, bucket, offset);

      }

      offset += sol_index[k];

    }

  }
}

template<bool Interleaved>
void Albany::MultiSTKFieldContainer<Interleaved>::saveResVector(const Epetra_Vector& res,
    stk_classic::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map) {

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType SFT;

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk_classic::mesh::BucketVector all_elements;
  stk_classic::mesh::get_buckets(sel, this->bulkData->buckets(this->metaData->node_rank()), all_elements);
  this->numNodes = node_map->NumMyElements(); // Needed for the getDOF function to work correctly
  // This is either numOwnedNodes or numOverlapNodes, depending on
  // which map is passed in

  for(stk_classic::mesh::BucketVector::const_iterator it = all_elements.begin() ; it != all_elements.end() ; ++it) {

    const stk_classic::mesh::Bucket& bucket = **it;

    int offset = 0;

    for(int k = 0; k < res_index.size(); k++) {

      if(res_index[k] == 1) { // Scalar

        SFT* field = this->metaData->template get_field<SFT>(res_vector_name[k]);
        this->saveVectorHelper(res, field, node_map, bucket, offset);

      }

      else {

        VFT* field = this->metaData->template get_field<VFT>(res_vector_name[k]);
        this->saveVectorHelper(res, field, node_map, bucket, offset);

      }

      offset += res_index[k];

    }

  }
}

template<bool Interleaved>
void Albany::MultiSTKFieldContainer<Interleaved>::transferSolutionToCoords() {

  const bool MultiSTKFieldContainer_transferSolutionToCoords_not_implemented = true;
  TEUCHOS_TEST_FOR_EXCEPT(MultiSTKFieldContainer_transferSolutionToCoords_not_implemented);
  //     this->copySTKField(solution_field, this->coordinates_field);

}
