//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_GenericSTKFieldContainer.hpp"

#include "Albany_Utils.hpp"
#include <stk_mesh/base/GetBuckets.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

template<bool Interleaved>
Albany::GenericSTKFieldContainer<Interleaved>::GenericSTKFieldContainer(
   const Teuchos::RCP<Teuchos::ParameterList>& params_,
   stk::mesh::fem::FEMMetaData* metaData_,
   stk::mesh::BulkData* bulkData_,
   const int neq_, 
   const int numDim_)
    : metaData(metaData_),
      bulkData(bulkData_),
      params(params_),
      neq(neq_),
      numDim(numDim_)
{
}

template<bool Interleaved>
Albany::GenericSTKFieldContainer<Interleaved>::~GenericSTKFieldContainer(){
}


template<bool Interleaved>
void
Albany::GenericSTKFieldContainer<Interleaved>::buildStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis){

  typedef typename AbstractSTKFieldContainer::QPScalarFieldType QPSFT;
  typedef typename AbstractSTKFieldContainer::QPVectorFieldType QPVFT;
  typedef typename AbstractSTKFieldContainer::QPTensorFieldType QPTFT;

  // Code to parse the vector of StateStructs and create STK fields
  for (std::size_t i=0; i<sis->size(); i++) {
    Albany::StateStruct& st = *((*sis)[i]);
    std::vector<int>& dim = st.dim;
    if (dim.size() == 2 && st.entity=="QuadPoint") {
      qpscalar_states.push_back(& metaData->declare_field< QPSFT >( st.name) );
      stk::mesh::put_field( *qpscalar_states.back() , metaData->element_rank(),
			    metaData->universal_part(), dim[1]);
//Debug
//      cout << "Allocating qps field name " << qpscalar_states.back()->name() << 
//            " size: (" << dim[0] << ", " << dim[1] << ")" <<endl;
#ifdef ALBANY_SEACAS
      if (st.output) stk::io::set_field_role(*qpscalar_states.back(), Ioss::Field::TRANSIENT);
#endif
    }
    else if (dim.size() == 3 && st.entity=="QuadPoint") {
      qpvector_states.push_back(& metaData->declare_field< QPVFT >( st.name) );
      // Multi-dim order is Fortran Ordering, so reversed here
      stk::mesh::put_field( *qpvector_states.back() , metaData->element_rank(),
			    metaData->universal_part(), dim[2], dim[1]);
//Debug
//      cout << "Allocating qpv field name " << qpvector_states.back()->name() << 
//            " size: (" << dim[0] << ", " << dim[1] << ", " << dim[2] << ")" <<endl;
#ifdef ALBANY_SEACAS
      if (st.output) stk::io::set_field_role(*qpvector_states.back(), Ioss::Field::TRANSIENT);
#endif
    }
    else if (dim.size() == 4 && st.entity=="QuadPoint") {
      qptensor_states.push_back(& metaData->declare_field< QPTFT >( st.name) );
      // Multi-dim order is Fortran Ordering, so reversed here
      stk::mesh::put_field( *qptensor_states.back() , metaData->element_rank(),
			    metaData->universal_part(), dim[3], dim[2], dim[1]);
//Debug
//      cout << "Allocating qpt field name " << qptensor_states.back()->name() << 
//            " size: (" << dim[0] << ", " << dim[1] << ", " << dim[2] << ", " << dim[3] << ")" <<endl;
#ifdef ALBANY_SEACAS
      if (st.output) stk::io::set_field_role(*qptensor_states.back(), Ioss::Field::TRANSIENT);
#endif
    }
    else if ( dim.size() == 1 && st.entity=="ScalarValue" ) {
      scalarValue_states.push_back(st.name);
    }
    else TEUCHOS_TEST_FOR_EXCEPT(dim.size() < 2 || dim.size()>4 || st.entity!="QuadPoint");

  }
}
  

template<bool Interleaved>
template<class T>
typename boost::disable_if< boost::is_same<T,Albany::AbstractSTKFieldContainer::ScalarFieldType>, void >::type
Albany::GenericSTKFieldContainer<Interleaved>::fillVectorHelper(Epetra_Vector &soln, 
             T *solution_field,
             const Teuchos::RCP<Epetra_Map>& node_map,
             const stk::mesh::Bucket & bucket, int offset){

    // Fill the result vector
    // Create a multidimensional array view of the
    // solution field data for this bucket of nodes.
    // The array is two dimensional ( Cartesian X NumberNodes )
    // and indexed by ( 0..2 , 0..NumberNodes-1 )

    stk::mesh::BucketArray<T>
        solution_array( *solution_field, bucket );

    const int num_vec_components = solution_array.dimension(0);
    const int num_nodes_in_bucket = solution_array.dimension(1);

    for (std::size_t i=0; i < num_nodes_in_bucket; i++)  {

//      const unsigned node_gid = bucket[i].identifier();
      const int node_gid = bucket[i].identifier() - 1;
      int node_lid = node_map->LID(node_gid);

      for (std::size_t j=0; j<num_vec_components; j++) {
        soln[getDOF(node_lid, offset + j)] = solution_array(j, i);

      }
    }
}

//Tpetra version of above 
template<bool Interleaved>
template<class T>
typename boost::disable_if< boost::is_same<T,Albany::AbstractSTKFieldContainer::ScalarFieldType>, void >::type
Albany::GenericSTKFieldContainer<Interleaved>::fillVectorHelperT(Tpetra_Vector &solnT, 
             T *solution_field,
             const Teuchos::RCP<const Tpetra_Map>& node_mapT,
             const stk::mesh::Bucket & bucket, int offset){
    
    // Fill the result vector
    // Create a multidimensional array view of the
    // solution field data for this bucket of nodes.
    // The array is two dimensional ( Cartesian X NumberNodes )
    // and indexed by ( 0..2 , 0..NumberNodes-1 )

    stk::mesh::BucketArray<T>
        solution_array( *solution_field, bucket );

    const int num_vec_components = solution_array.dimension(0);
    const int num_nodes_in_bucket = solution_array.dimension(1);

    for (std::size_t i=0; i < num_nodes_in_bucket; i++)  {

//      const unsigned node_gid = bucket[i].identifier();
      const int node_gid = bucket[i].identifier() - 1;
      int node_lid = node_mapT->getLocalElement(node_gid);

      for (std::size_t j=0; j<num_vec_components; j++) {
        solnT.replaceLocalValue(getDOF(node_lid, offset+j), solution_array(j, i)); 

      }
    }
}

// Specialization for ScalarFieldType

template<bool Interleaved>
void Albany::GenericSTKFieldContainer<Interleaved>::fillVectorHelper(Epetra_Vector &soln, 
             ScalarFieldType *solution_field,
             const Teuchos::RCP<Epetra_Map>& node_map,
             const stk::mesh::Bucket & bucket, int offset){

    // Fill the result vector
    // Create a multidimensional array view of the
    // solution field data for this bucket of nodes.
    // The array is two dimensional ( Cartesian X NumberNodes )
    // and indexed by ( 0..2 , 0..NumberNodes-1 )

    stk::mesh::BucketArray<ScalarFieldType>
        solution_array( *solution_field, bucket );

    const int num_nodes_in_bucket = solution_array.dimension(0);

    for (std::size_t i=0; i < num_nodes_in_bucket; i++)  {

//      const unsigned node_gid = bucket[i].identifier();
      const int node_gid = bucket[i].identifier() - 1;
      int node_lid = node_map->LID(node_gid);

      soln[getDOF(node_lid, offset)] = solution_array(i);

    }
}

//Tpetra version of above
template<bool Interleaved>
void Albany::GenericSTKFieldContainer<Interleaved>::fillVectorHelperT(Tpetra_Vector &solnT, 
             ScalarFieldType *solution_field,
             const Teuchos::RCP<const Tpetra_Map>& node_mapT,
             const stk::mesh::Bucket & bucket, int offset){
    
    // Fill the result vector
    // Create a multidimensional array view of the
    // solution field data for this bucket of nodes.
    // The array is two dimensional ( Cartesian X NumberNodes )
    // and indexed by ( 0..2 , 0..NumberNodes-1 )

    stk::mesh::BucketArray<ScalarFieldType>
        solution_array( *solution_field, bucket );

    const int num_nodes_in_bucket = solution_array.dimension(0);

    for (std::size_t i=0; i < num_nodes_in_bucket; i++)  {

//      const unsigned node_gid = bucket[i].identifier();
      const int node_gid = bucket[i].identifier() - 1;
      int node_lid = node_mapT->getLocalElement(node_gid);
 
      solnT.replaceLocalValue(getDOF(node_lid, offset), solution_array(i)); 

    }
}

template<bool Interleaved>
template<class T>
typename boost::disable_if< boost::is_same<T,Albany::AbstractSTKFieldContainer::ScalarFieldType>, void >::type
Albany::GenericSTKFieldContainer<Interleaved>::saveVectorHelper(const Epetra_Vector &soln, 
             T *solution_field,
             const Teuchos::RCP<Epetra_Map>& node_map,
             const stk::mesh::Bucket & bucket, int offset){

    // Fill the result vector
    // Create a multidimensional array view of the
    // solution field data for this bucket of nodes.
    // The array is two dimensional ( Cartesian X NumberNodes )
    // and indexed by ( 0..2 , 0..NumberNodes-1 )

    stk::mesh::BucketArray<T>
        solution_array( *solution_field, bucket );

    const int num_vec_components = solution_array.dimension(0);
    const int num_nodes_in_bucket = solution_array.dimension(1);

    for (std::size_t i=0; i < num_nodes_in_bucket; i++)  {

//      const unsigned node_gid = bucket[i].identifier();
      const int node_gid = bucket[i].identifier() - 1;
      int node_lid = node_map->LID(node_gid);

      for (std::size_t j=0; j<num_vec_components; j++) {
        solution_array(j, i) = soln[getDOF(node_lid, offset + j)];

      }
    }
}

// Specialization for ScalarFieldType
template<bool Interleaved>
void Albany::GenericSTKFieldContainer<Interleaved>::saveVectorHelper(const Epetra_Vector &soln, 
             ScalarFieldType *solution_field,
             const Teuchos::RCP<Epetra_Map>& node_map,
             const stk::mesh::Bucket & bucket, int offset){

    // Fill the result vector
    // Create a multidimensional array view of the
    // solution field data for this bucket of nodes.
    // The array is two dimensional ( Cartesian X NumberNodes )
    // and indexed by ( 0..2 , 0..NumberNodes-1 )

    stk::mesh::BucketArray<ScalarFieldType>
        solution_array( *solution_field, bucket );

    const int num_nodes_in_bucket = solution_array.dimension(0);

    for (std::size_t i=0; i < num_nodes_in_bucket; i++)  {

//      const unsigned node_gid = bucket[i].identifier();
      const int node_gid = bucket[i].identifier() - 1;
      int node_lid = node_map->LID(node_gid);

      solution_array(i) = soln[getDOF(node_lid, offset)];

    }
}

