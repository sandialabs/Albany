//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP
#define ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP

#include "Albany_GenericSTKFieldContainer.hpp"

namespace Albany {

template <bool Interleaved>
class OrdinarySTKFieldContainer : public GenericSTKFieldContainer<Interleaved>
{
 public:
  OrdinarySTKFieldContainer(
      const Teuchos::RCP<Teuchos::ParameterList>&               params_,
      const Teuchos::RCP<stk::mesh::MetaData>&                  metaData_,
      const Teuchos::RCP<stk::mesh::BulkData>&                  bulkData_,
      const int                                                 neq_,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const int                                                 numDim_,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis, 
      const int                                                 num_params);

  ~OrdinarySTKFieldContainer() = default;

  bool
  hasResidualField() const
  {
    return (residual_field != NULL);
  }

  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*>
  getSolutionFieldArray()
  {
    return solution_field;
  }

  AbstractSTKFieldContainer::VectorFieldType*
  getSolutionField()
  {
    return solution_field[0];
  };

#if defined(ALBANY_DTK)
  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*>
  getSolutionFieldDTKArray()
  {
    return solution_field_dtk;
  };

  AbstractSTKFieldContainer::VectorFieldType*
  getSolutionFieldDTK()
  {
    return solution_field_dtk[0];
  };
#endif

  void
  fillSolnVector(
      Thyra_Vector&                                soln,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  fillVector(
      Thyra_Vector&                                field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);
  void
  fillSolnMultiVector(
      Thyra_MultiVector&                           soln,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveVector(
      const Thyra_Vector&                          field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);
  void
  saveSolnVector(
      const Thyra_Vector&                          soln, 
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      const Thyra_Vector&                          soln_dot,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      const Thyra_Vector&                          soln_dot,
      const Thyra_Vector&                          soln_dotdot,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveResVector(
      const Thyra_Vector&                          res,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);
  void
  saveSolnMultiVector(
      const Thyra_MultiVector&                     soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs);

  void
  transferSolutionToCoords();

 private:
  void
  fillVectorImpl(
      Thyra_Vector&                                field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);
  void
  saveVectorImpl(
      const Thyra_Vector&                          field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager);

  void
  initializeSTKAdaptation();

  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*> solution_field;
  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*>
                                              solution_field_dtk;
  Teuchos::Array<AbstractSTKFieldContainer::VectorFieldType*>
                                              solution_field_dxdp;
  AbstractSTKFieldContainer::VectorFieldType* residual_field;

  int num_params{0}; 
};

}  // namespace Albany

#endif  // ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP
