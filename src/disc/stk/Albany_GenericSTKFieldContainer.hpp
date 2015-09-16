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
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>

#include <boost/utility.hpp>
#include <boost/type_traits/is_same.hpp>


namespace Albany {

template<bool Interleaved>

class GenericSTKFieldContainer : public AbstractSTKFieldContainer {

  public:

    GenericSTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                             const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
                             const int neq_,
                             const int numDim_);

    virtual ~GenericSTKFieldContainer();

    // Add StateStructs to the list of stored ones
    void addStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis);

  protected:

    // non-interleaved version
    inline int getDOF(const int inode, const int eq) const {
      return inode + numNodes * eq;
    }

    // Use boost to provide explicit specialization - Tpetra
    template<class T>
    typename boost::disable_if< boost::is_same<T,ScalarFieldType>, void >::type
    fillVectorHelperT(Tpetra_Vector& solnT,
                      T* solution_field,
                      const Teuchos::RCP<const Tpetra_Map>& node_mapT,
                      const stk::mesh::Bucket& bucket, int offset);

    void fillVectorHelperT(Tpetra_Vector& solnT,
                           ScalarFieldType* solution_field,
                           const Teuchos::RCP<const Tpetra_Map>& node_mapT,
                           const stk::mesh::Bucket& bucket, int offset);

    template<class T>
    typename boost::disable_if< boost::is_same<T, ScalarFieldType>, void >::type
    saveVectorHelperT(const Tpetra_Vector& solnT,
                     T* solution_field,
                     const Teuchos::RCP<const Tpetra_Map>& node_mapT,
                     const stk::mesh::Bucket& bucket, int offset);

    void saveVectorHelperT(const Tpetra_Vector& solnT,
                          ScalarFieldType* solution_field,
                          const Teuchos::RCP<const Tpetra_Map>& node_mapT,
                          const stk::mesh::Bucket& bucket, int offset);

#if defined(ALBANY_EPETRA)
    // Use boost to provide explicit specialization
    template<class T>
    typename boost::disable_if< boost::is_same<T, ScalarFieldType>, void >::type
    fillVectorHelper(Epetra_Vector& soln,
                     T* solution_field,
                     const Teuchos::RCP<Epetra_Map>& node_map,
                     const stk::mesh::Bucket& bucket, int offset);

    void fillVectorHelper(Epetra_Vector& soln,
                          ScalarFieldType* solution_field,
                          const Teuchos::RCP<Epetra_Map>& node_map,
                          const stk::mesh::Bucket& bucket, int offset);

    template<class T>
    typename boost::disable_if< boost::is_same<T, ScalarFieldType>, void >::type
    fillVectorHelper(Epetra_Vector& field_vector,
                     T* field, const Teuchos::RCP<Epetra_Map>& node_map,
                     const stk::mesh::Bucket& bucket, const NodalDOFManager& nodalDofManager, int offset=0);

    void fillVectorHelper(Epetra_Vector& field_vector,
                          ScalarFieldType* field, const Teuchos::RCP<Epetra_Map>& node_map,
                          const stk::mesh::Bucket& bucket, const NodalDOFManager& nodalDofManager, int offset=0);

    template<class T>
    typename boost::disable_if< boost::is_same<T, ScalarFieldType>, void >::type
    saveVectorHelper(const Epetra_Vector& field_vector,
                     T* field, const Teuchos::RCP<Epetra_Map>& node_map,
                     const stk::mesh::Bucket& bucket, const NodalDOFManager& nodalDofManager, int offset=0);

    void saveVectorHelper(const Epetra_Vector& field_vector,
                          ScalarFieldType* field, const Teuchos::RCP<Epetra_Map>& node_map,
                          const stk::mesh::Bucket& bucket, const NodalDOFManager& nodalDofManager, int offset=0);

    // Use boost to provide explicit specialization
    template<class T>
    typename boost::disable_if< boost::is_same<T, ScalarFieldType>, void >::type
    saveVectorHelper(const Epetra_Vector& soln,
                     T* solution_field,
                     const Teuchos::RCP<Epetra_Map>& node_map,
                     const stk::mesh::Bucket& bucket, int offset);

    void saveVectorHelper(const Epetra_Vector& soln,
                          ScalarFieldType* solution_field,
                          const Teuchos::RCP<Epetra_Map>& node_map,
                          const stk::mesh::Bucket& bucket, int offset);
#endif

    // Convenience function to copy one field's contents to another
    template<class T>
    typename boost::disable_if< boost::is_same<T, ScalarFieldType>, void >::type
    copySTKField(const T* source, T* target);

    // Specialization for ScalarFieldType
    void copySTKField(const ScalarFieldType* source, ScalarFieldType* target);

    Teuchos::RCP<stk::mesh::MetaData> metaData;
    Teuchos::RCP<Teuchos::ParameterList> params;

    int numNodes; // used to implement getDOF function when ! interleaved
    int neq;
    int numDim;

};

// interleaved version
template<> inline int GenericSTKFieldContainer<true>::getDOF(const int inode, const int eq) const {
  return inode * neq + eq;
}

} // namespace Albany



// Define macro for explicit template instantiation
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  template class name<false>;
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name) \
  template class name<true>;
#if defined(ALBANY_EPETRA)
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVH(class_name, value, arg_type) \
  template void class_name<value>::saveVectorHelper( \
             const Epetra_Vector &soln, \
             arg_type *solution_field, \
             const Teuchos::RCP<Epetra_Map>& node_map, \
             const stk::mesh::Bucket & bucket, int offset);
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVHN(class_name, value, arg_type) \
  template void class_name<value>::saveVectorHelper( \
             const Epetra_Vector &field_vector, \
             arg_type *solution_field, \
             const Teuchos::RCP<Epetra_Map>& node_map, \
             const stk::mesh::Bucket & bucket, \
             const NodalDOFManager& nodalDofManager, int offset);
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVH(class_name, value, arg_type) \
  template void class_name<value>::fillVectorHelper( \
          Epetra_Vector &soln,  \
          arg_type *solution_field, \
          const Teuchos::RCP<Epetra_Map>& node_map,  \
          const stk::mesh::Bucket & bucket, int offset);
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVHN(class_name, value, arg_type) \
  template void class_name<value>::fillVectorHelper( \
          Epetra_Vector& field_vector, \
          arg_type* field, \
          const Teuchos::RCP<Epetra_Map>& node_map, \
          const stk::mesh::Bucket& bucket, \
          const NodalDOFManager& nodalDofManager, int offset);
#endif // ALBANY_EPETRA
//Tpetra version of some of the above
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVHT(class_name, value, arg_type) \
  template void class_name<value>::saveVectorHelperT( \
             const Tpetra_Vector &solnT, \
             arg_type *solution_field, \
             const Teuchos::RCP<const Tpetra_Map>& node_mapT, \
             const stk::mesh::Bucket & bucket, int offset);
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVHT(class_name, value, arg_type) \
  template void class_name<value>::fillVectorHelperT( \
          Tpetra_Vector &solnT,  \
          arg_type *solution_field, \
          const Teuchos::RCP<const Tpetra_Map>& node_mapT,  \
          const stk::mesh::Bucket & bucket, int offset);
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_CSTKF(class_name, value, arg_type) \
  template void class_name<value>::copySTKField( \
          const arg_type *source_field, \
          arg_type *target_field);

#if defined(ALBANY_EPETRA)
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS(name) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVH(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVH(name, false, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVH(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVH(name, false, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVHN(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVHN(name, false, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVHN(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVHN(name, false, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVHT(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVHT(name, false, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVHT(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVHT(name, false, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_CSTKF(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_CSTKF(name, false, AbstractSTKFieldContainer::VectorFieldType)
#else
#define STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS(name) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVHT(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_SVHT(name, false, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVHT(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_FVHT(name, false, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_CSTKF(name, true, AbstractSTKFieldContainer::VectorFieldType) \
  STKFIELDCONTAINER_INSTANTIATE_TEMPLATE_FUNCTION_CSTKF(name, false, AbstractSTKFieldContainer::VectorFieldType)
#endif // ALBANY_EPETRA
#endif // ALBANY_GENERICSTKFIELDCONT_HPP
