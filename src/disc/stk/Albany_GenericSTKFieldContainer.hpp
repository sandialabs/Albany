//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GENERICSTKFIELDCONT_HPP
#define ALBANY_GENERICSTKFIELDCONT_HPP

#include "Albany_AbstractSTKFieldContainer.hpp"
#include "Teuchos_ParameterList.hpp"


// Start of STK stuff
#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/fem/FEMMetaData.hpp>

#include <boost/utility.hpp>
#include <boost/type_traits/is_same.hpp>


namespace Albany {

template<bool Interleaved>

  class GenericSTKFieldContainer : public AbstractSTKFieldContainer {

    public:

    GenericSTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
      stk::mesh::fem::FEMMetaData* metaData_,
      stk::mesh::BulkData* bulkData_,
      const int neq_, 
      const int numDim_);

    virtual ~GenericSTKFieldContainer();

    protected:

       // non-interleaved version 
       inline int getDOF(const int inode, const int eq) const
          { return inode + numNodes*eq; }

      // Build StateStructs
      void buildStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis);


      // Use boost to provide explicit specialization 
      template<class T>
        typename boost::disable_if< boost::is_same<T,ScalarFieldType>, void >::type
         fillVectorHelper(Epetra_Vector &soln, 
             T *solution_field,
             const Teuchos::RCP<Epetra_Map>& node_map, 
             const stk::mesh::Bucket & bucket, int offset);

      void fillVectorHelper(Epetra_Vector &soln, 
          ScalarFieldType *solution_field,
          const Teuchos::RCP<Epetra_Map>& node_map, 
          const stk::mesh::Bucket & bucket, int offset);

      // Use boost to provide explicit specialization 
      template<class T>
        typename boost::disable_if< boost::is_same<T,ScalarFieldType>, void >::type
          saveVectorHelper(const Epetra_Vector &soln, 
            T *solution_field,
            const Teuchos::RCP<Epetra_Map>& node_map,
            const stk::mesh::Bucket & bucket, int offset);

       void saveVectorHelper(const Epetra_Vector &soln, 
           ScalarFieldType *solution_field,
           const Teuchos::RCP<Epetra_Map>& node_map,
           const stk::mesh::Bucket & bucket, int offset);

       stk::mesh::fem::FEMMetaData* metaData;
       stk::mesh::BulkData* bulkData;
       Teuchos::RCP<Teuchos::ParameterList> params;

       int numNodes; // used to implement getDOF function when ! interleaved
       int neq;
       int numDim;

  };

  // interleaved version
  template<> inline int GenericSTKFieldContainer<true>::getDOF(const int inode, const int eq) const
                        { return inode*neq + eq; }

} // namespace Albany



// Define macro for explicit template instantiation
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  template class name<false>;
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name) \
  template class name<true>;
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVH(class_name, value, arg_type) \
  template void class_name<value>::saveVectorHelper( \
             const Epetra_Vector &soln, \
             arg_type *solution_field, \
             const Teuchos::RCP<Epetra_Map>& node_map, \
             const stk::mesh::Bucket & bucket, int offset);
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVH(class_name, value, arg_type) \
  template void class_name<value>::fillVectorHelper( \
          Epetra_Vector &soln,  \
          arg_type *solution_field, \
          const Teuchos::RCP<Epetra_Map>& node_map,  \
          const stk::mesh::Bucket & bucket, int offset);



#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS(name) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVH(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVH(name, false, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVH(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVH(name, false, AbstractSTKFieldContainer::VectorFieldType) 


#endif
