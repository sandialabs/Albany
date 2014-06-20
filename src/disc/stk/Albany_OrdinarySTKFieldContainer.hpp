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
                              stk_classic::mesh::fem::FEMMetaData* metaData_,
                              stk_classic::mesh::BulkData* bulkData_,
                              const int neq_,
                              const AbstractFieldContainer::FieldContainerRequirements& req,
                              const int numDim_,
                              const Teuchos::RCP<Albany::StateInfoStruct>& sis);

    ~OrdinarySTKFieldContainer();

    bool hasResidualField(){ return (residual_field != NULL); }
    bool hasSurfaceHeightField(){ return buildSurfaceHeight; }
    bool hasTemperatureField(){ return buildTemperature; }
    bool hasBasalFrictionField(){ return buildBasalFriction; }
    bool hasThicknessField(){ return buildThickness; }
    bool hasFlowFactorField(){ return buildFlowFactor; }
    bool hasSurfaceVelocityField(){ return buildSurfaceVelocity; }
    bool hasVelocityRMSField(){ return buildVelocityRMS; }
    bool hasSphereVolumeField(){ return buildSphereVolume; }

    AbstractSTKFieldContainer::VectorFieldType* getSolutionField(){ return solution_field; };

    void fillSolnVector(Epetra_Vector& soln, stk_classic::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
    void saveSolnVector(const Epetra_Vector& soln, stk_classic::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);
    void saveResVector(const Epetra_Vector& res, stk_classic::mesh::Selector& sel, const Teuchos::RCP<Epetra_Map>& node_map);

    void transferSolutionToCoords();

  private:

    void initializeSTKAdaptation();

    bool buildSurfaceHeight;
    bool buildTemperature;
    bool buildBasalFriction;
    bool buildThickness;
    bool buildFlowFactor;
    bool buildSurfaceVelocity;
    bool buildVelocityRMS;
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
