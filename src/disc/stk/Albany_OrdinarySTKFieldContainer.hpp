//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ORDINARYSTKFIELDCONT_HPP
#define ALBANY_ORDINARYSTKFIELDCONT_HPP

#include "Albany_GenericSTKFieldContainer.hpp"

namespace Albany {

template<bool Interleaved>

class OrdinarySTKFieldContainer : public GenericSTKFieldContainer<Interleaved> {

  public:

    OrdinarySTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                              const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
                              const int neq_,
                              const AbstractFieldContainer::FieldContainerRequirements& req,
                              const int numDim_,
                              const Teuchos::RCP<Albany::StateInfoStruct>& sis);

    ~OrdinarySTKFieldContainer();

    bool hasResidualField(){ return (residual_field != NULL); }
    bool hasSphereVolumeField(){ return buildSphereVolume; }

    AbstractSTKFieldContainer::VectorFieldType* getSolutionField(){ return solution_field; };

#if defined(ALBANY_EPETRA)
    void fillSolnVector(Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
#endif
    void fillSolnVectorT(Tpetra_Vector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT);
#if defined(ALBANY_EPETRA)
    void fillVector(Epetra_Vector& field_vector, const std::string& field_name, stk::mesh::Selector& field_selection,
                    const Teuchos::RCP<Epetra_Map>& field_node_map, const NodalDOFManager& nodalDofManager);

    void saveVector(const Epetra_Vector& field_vector, const std::string& field_name, stk::mesh::Selector& field_selection,
                        const Teuchos::RCP<Epetra_Map>& field_node_map, const NodalDOFManager& nodalDofManager);

    void saveSolnVector(const Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
#endif
    //Tpetra version of above
    void saveSolnVectorT(const Tpetra_Vector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT);
#if defined(ALBANY_EPETRA)
    void saveResVector(const Epetra_Vector& res, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
#endif
    void saveResVectorT(const Tpetra_Vector& res, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_map);

    void transferSolutionToCoords();

  private:

    void initializeSTKAdaptation();

    bool buildSphereVolume;

    AbstractSTKFieldContainer::VectorFieldType* solution_field;
    AbstractSTKFieldContainer::VectorFieldType* residual_field;

};

} // namespace Albany



// Define macro for explicit template instantiation
#define ORDINARYSTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  template class name<false>;
#define ORDINARYSTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name) \
  template class name<true>;

#define ORDINARYSTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS(name) \
  ORDINARYSTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  ORDINARYSTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name)


#endif
