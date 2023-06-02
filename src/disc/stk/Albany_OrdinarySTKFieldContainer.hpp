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
      const int                                                 numDim_,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
      const int                                                 num_params);

  OrdinarySTKFieldContainer(
      const Teuchos::RCP<Teuchos::ParameterList>&               params_,
      const Teuchos::RCP<stk::mesh::MetaData>&                  metaData_,
      const Teuchos::RCP<stk::mesh::BulkData>&                  bulkData_,
      const int                                                 neq_,
      const int                                                 numDim_,
      const Teuchos::RCP<Albany::StateInfoStruct>&              sis,
      const int                                                 num_params);

  ~OrdinarySTKFieldContainer() = default;

  bool
  hasResidualField() const
  {
    return (residual_field != NULL);
  }

  Teuchos::Array<AbstractSTKFieldContainer::STKFieldType*>
  getSolutionFieldArray()
  {
    return solution_field;
  }

  AbstractSTKFieldContainer::STKFieldType*
  getSolutionField()
  {
    return solution_field[0];
  };

#if defined(ALBANY_DTK)
  Teuchos::Array<AbstractSTKFieldContainer::STKFieldType*>
  getSolutionFieldDTKArray()
  {
    return solution_field_dtk;
  };

  AbstractSTKFieldContainer::STKFieldType*
  getSolutionFieldDTK()
  {
    return solution_field_dtk[0];
  };
#endif

  void fillSolnVector (Thyra_Vector&        soln,
                       const dof_mgr_ptr_t& soln_dof_mgr,
                       const bool           overlapped);

  void fillVector (Thyra_Vector&        field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped);

  void fillSolnMultiVector (Thyra_MultiVector&   soln,
                            const dof_mgr_ptr_t& soln_dof_mgr,
                            const bool           overlapped);

  void saveVector (const Thyra_Vector&  field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped);

  void saveSolnVector (const Thyra_Vector& soln,
                       const mv_ptr_t&     soln_dxdp,
                       const dof_mgr_ptr_t& sol_dof_mgr,
                       const bool           overlapped);

  void saveSolnVector (const Thyra_Vector&  soln,
                       const mv_ptr_t&      soln_dxdp,
                       const Thyra_Vector&  soln_dot,
                       const dof_mgr_ptr_t& sol_dof_mgr,
                       const bool           overlapped);

  void saveSolnVector (const Thyra_Vector&  soln,
                       const mv_ptr_t&      soln_dxdp,
                       const Thyra_Vector&  soln_dot,
                       const Thyra_Vector&  soln_dotdot,
                       const dof_mgr_ptr_t& sol_dof_mgr,
                       const bool           overlapped);

  void saveResVector (const Thyra_Vector&  res,
                      const dof_mgr_ptr_t& dof_mgr,
                      const bool           overlapped);

  void saveSolnMultiVector (const Thyra_MultiVector& soln,
                            const mv_ptr_t&          soln_dxdp,
                            const dof_mgr_ptr_t&     sol_dof_mgr,
                            const bool               overlapped);

  void transferSolutionToCoords();


 private:
  void fillVectorImpl (Thyra_Vector&        field_vector,
                       const std::string&   field_name,
                       const dof_mgr_ptr_t& field_dof_mgr,
                       const bool           overlapped);

  void saveVectorImpl (const Thyra_Vector&  field_vector,
                       const std::string&   field_name,
                       const dof_mgr_ptr_t& field_dof_mgr,
                       const bool           overlapped);

  void initializeProcRankField();

  Teuchos::Array<AbstractSTKFieldContainer::STKFieldType*> solution_field;
  Teuchos::Array<AbstractSTKFieldContainer::STKFieldType*>
                                              solution_field_dtk;
  Teuchos::Array<AbstractSTKFieldContainer::STKFieldType*>
                                              solution_field_dxdp;
  AbstractSTKFieldContainer::STKFieldType* residual_field;
};

}  // namespace Albany

#endif  // ALBANY_ORDINARY_STK_FIELD_CONTAINER_HPP
