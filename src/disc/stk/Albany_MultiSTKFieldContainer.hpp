//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra ifdef'ed out if ALBANY_EPETRA_EXE turned off

#ifndef ALBANY_MULTISTKFIELDCONT_HPP
#define ALBANY_MULTISTKFIELDCONT_HPP

#include "Albany_GenericSTKFieldContainer.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

template<bool Interleaved>

class MultiSTKFieldContainer : public GenericSTKFieldContainer<Interleaved> {

  public:

    MultiSTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                           const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
                           const int neq_,
                           const AbstractFieldContainer::FieldContainerRequirements& req,
                           const int numDim_,
                           const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                           const Teuchos::Array<std::string>& solution_vector,
                           const Teuchos::Array<std::string>& residual_vector);

    ~MultiSTKFieldContainer();

    bool hasResidualField(){ return haveResidual; }
    bool hasSphereVolumeField(){ return buildSphereVolume; }

#ifdef ALBANY_EPETRA
    void fillSolnVector(Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
#endif
    void fillSolnVectorT(Tpetra_Vector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT);
#ifdef ALBANY_EPETRA
    void saveSolnVector(const Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
#endif
    //Tpetra version of above
    void saveSolnVectorT(const Tpetra_Vector& solnT, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_mapT);
#ifdef ALBANY_EPETRA
    void saveResVector(const Epetra_Vector& res, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
#endif
    void saveResVectorT(const Tpetra_Vector& res, stk::mesh::Selector& sel, const Teuchos::RCP<const Tpetra_Map>& node_map);

#ifdef ALBANY_EPETRA
    void fillVector(Epetra_Vector& field_vector, const std::string&  field_name,
        stk::mesh::Selector& field_selection, const Teuchos::RCP<Epetra_Map>& field_node_map, const NodalDOFManager& nodalDofManager);

    void saveVector(const Epetra_Vector& field_vector, const std::string&  field_name,
        stk::mesh::Selector& field_selection, const Teuchos::RCP<Epetra_Map>& field_node_map, const NodalDOFManager& nodalDofManager);
#endif

    void transferSolutionToCoords();

  private:

    void initializeSTKAdaptation();

    bool haveResidual;

    bool buildSphereVolume;

    // Containers for residual and solution

    std::vector<std::string> sol_vector_name;
    std::vector<int> sol_index;

    std::vector<std::string> res_vector_name;
    std::vector<int> res_index;

};

} // namespace Albany



// Define macro for explicit template instantiation
#define MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  template class name<false>;
#define MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name) \
  template class name<true>;

#define MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS(name) \
  MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_NONINTERLEAVED(name) \
  MULTISTKFIELDCONTAINER_INSTANTIATE_TEMPLATE_CLASS_INTERLEAVED(name)


#endif
