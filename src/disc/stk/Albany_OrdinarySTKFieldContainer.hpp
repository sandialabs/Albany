//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP
#define ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP

#include "Albany_GenericSTKFieldContainer.hpp"

namespace Albany {

class OrdinarySTKFieldContainer : public GenericSTKFieldContainer
{
 public:
  OrdinarySTKFieldContainer(
      const Teuchos::RCP<Teuchos::ParameterList>&               params_,
      const Teuchos::RCP<stk::mesh::MetaData>&                  metaData_,
      const Teuchos::RCP<stk::mesh::BulkData>&                  bulkData_,
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const DiscType                                            interleaved_,
      const int                                                 numDim_,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
      const int                                                 num_params);

  OrdinarySTKFieldContainer(
      const Teuchos::RCP<Teuchos::ParameterList>&               params_,
      const Teuchos::RCP<stk::mesh::MetaData>&                  metaData_,
      const Teuchos::RCP<stk::mesh::BulkData>&                  bulkData_,
      const DiscType                                            interleaved_,
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

  void fillSolnVector (Thyra_Vector& solution,
                       const DOF&    solution_dof);

  void fillVector (Thyra_Vector&      field_vector,
                   const std::string& field_name,
                   const DOF&         field_dof);

  void fillSolnMultiVector (Thyra_MultiVector& solution,
                            const DOF&         solution_dof);

  void saveVector (const Thyra_Vector&  field_vector,
                   const std::string&   field_name,
                   const DOF&           field_dof);

  void saveSolnVector (const Thyra_Vector&  solution,
                       const cmv_ptr_t&     solution_dxdp,
                       const DOF&           solution_dof);

  void saveSolnVector (const Thyra_Vector&  solution,
                       const cmv_ptr_t&     solution_dxdp,
                       const Thyra_Vector&  solution_dot,
                       const DOF&           solution_dof);

  void saveSolnVector (const Thyra_Vector&  solution,
                       const cmv_ptr_t&     solution_dxdp,
                       const Thyra_Vector&  solution_dot,
                       const Thyra_Vector&  solution_dotdot,
                       const DOF&           solution_dof);

  void saveResVector (const Thyra_Vector& residual,
                      const DOF&          solution_dof);

  void saveSolnMultiVector (const Thyra_MultiVector&  solution,
                            const cmv_ptr_t&          solution_dxdp,
                            const DOF&                solution_dof);

  void transferSolutionToCoords();


 private:
  void fillVectorImpl (Thyra_Vector&      field_vector,
                       const std::string& field_name,
                       const DOF&         field_dof);

  void saveVectorImpl (const Thyra_Vector&  field_vector,
                       const std::string&   field_name,
                       const DOF&           field_dof);

  void initializeProcRankField();


  Teuchos::Array<VectorFieldType*>  solution_field;
  Teuchos::Array<VectorFieldType*>  solution_field_dtk;
  Teuchos::Array<VectorFieldType*>  solution_field_dxdp;

  VectorFieldType*                  residual_field;
};

}  // namespace Albany

#endif  // ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP
