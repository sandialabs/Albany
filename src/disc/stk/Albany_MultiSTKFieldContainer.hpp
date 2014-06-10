//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MULTISTKFIELDCONT_HPP
#define ALBANY_MULTISTKFIELDCONT_HPP

#include "Albany_GenericSTKFieldContainer.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

template<bool Interleaved>

class MultiSTKFieldContainer : public GenericSTKFieldContainer<Interleaved> {

  public:

    MultiSTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                           stk::mesh::fem::FEMMetaData* metaData_,
                           stk::mesh::BulkData* bulkData_,
                           const int neq_,
                           const AbstractFieldContainer::FieldContainerRequirements& req,
                           const int numDim_,
                           const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                           const Teuchos::Array<std::string>& solution_vector,
                           const Teuchos::Array<std::string>& residual_vector);

    ~MultiSTKFieldContainer();

    bool hasResidualField(){ return haveResidual; }
    bool hasSurfaceHeightField(){ return false; }
    bool hasTemperatureField(){ return false; }
    bool hasBasalFrictionField(){ return false; }
    bool hasThicknessField(){ return false; }
    bool hasFlowFactorField(){ return false; }
    bool hasSurfaceVelocityField(){ return false; }
    bool hasVelocityRMSField(){ return false; }
    bool hasSphereVolumeField(){ return buildSphereVolume; }

    void fillSolnVector(Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
    void saveSolnVector(const Epetra_Vector& soln, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
    void saveResVector(const Epetra_Vector& res, stk::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);

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
