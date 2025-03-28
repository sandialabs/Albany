//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MULTI_STK_FIELD_CONTAINER_HPP
#define ALBANY_MULTI_STK_FIELD_CONTAINER_HPP

#include "Albany_GenericSTKFieldContainer.hpp"
#include "Teuchos_Array.hpp"

namespace Albany {

class MultiSTKFieldContainer : public GenericSTKFieldContainer
{
public:
  MultiSTKFieldContainer(
      const Teuchos::RCP<Teuchos::ParameterList>&     params_,
      const Teuchos::RCP<stk::mesh::MetaData>&        metaData_,
      const Teuchos::RCP<stk::mesh::BulkData>&        bulkData_,
      const int                                       numDim_,
      const Teuchos::RCP<Albany::StateInfoStruct>&    sis,
      const int                                       num_params);
 
  MultiSTKFieldContainer(
      const Teuchos::RCP<Teuchos::ParameterList>&           params_,
      const Teuchos::RCP<stk::mesh::MetaData>&              metaData_,
      const Teuchos::RCP<stk::mesh::BulkData>&              bulkData_,
      const int                                             neq_,
      const int                                             numDim_,
      const Teuchos::RCP<Albany::StateInfoStruct>&          sis,
      const Teuchos::Array<Teuchos::Array<std::string>>&    solution_vector,
      const int                                             num_params);

  ~MultiSTKFieldContainer() = default;


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

  void fillSolnSensitivity (Thyra_MultiVector&   dxdp,
                            const dof_mgr_ptr_t& soln_dof_mgr,
                            const bool           overlapped);

  void transferSolutionToCoords();

private:
  void fillVectorImpl (      Thyra_Vector&     field_vector,
                       const std::string&      field_name,
                       const dof_mgr_ptr_t&    field_dof_mgr,
                       const bool              overlapped,
                       const std::vector<int>& components = {});

  void saveVectorImpl (const Thyra_Vector&     field_vector,
                       const std::string&      field_name,
                       const dof_mgr_ptr_t&    field_dof_mgr,
                       const bool              overlapped,
                       const std::vector<int>& components = {});

  void initializeProcRankField();

  // Containers for residual and solution

  Teuchos::Array<Teuchos::Array<std::string>> sol_vector_name;
  Teuchos::Array<Teuchos::Array<int>>         sol_index;

  Teuchos::Array<std::string> res_vector_name;
  Teuchos::Array<int>         res_index;
};

}  // namespace Albany

#endif // ALBANY_MULTI_STK_FIELD_CONTAINER_HPP
