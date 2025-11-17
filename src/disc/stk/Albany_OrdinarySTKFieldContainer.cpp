//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_Macros.hpp"
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_StringUtils.hpp"

// Start of STK stuff
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

namespace Albany {

OrdinarySTKFieldContainer::
OrdinarySTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                          const Teuchos::RCP<stk::mesh::MetaData>&    metaData_,
                          const Teuchos::RCP<stk::mesh::BulkData>&    bulkData_,
                          const int                                   num_params_,
                          const bool                                  set_geo_fields_meta_data)
 : GenericSTKFieldContainer(params_,metaData_,bulkData_)
 , num_params (num_params_)
{
  if (set_geo_fields_meta_data) {
    setGeometryFieldsMetadata ();
  }
}

void OrdinarySTKFieldContainer::
setSolutionFieldsMetadata (int neq)
{
  this->solutionFieldContainer = true;

  save_solution_field = params->get("Save Solution Field", true);
  if (not save_solution_field) {
    return;
  }

  using strvec_t = std::vector<std::string>;

  strvec_t sol_tag_name = {"Exodus Solution Name",
                           "Exodus SolutionDot Name",
                           "Exodus SolutionDotDot Name"};

  strvec_t sol_id_name = {"solution",
                          "solution_dot",
                          "solution_dotdot"};

  constexpr auto ELEM_RANK = stk::topology::ELEM_RANK;
  constexpr auto NODE_RANK = stk::topology::NODE_RANK;

  int num_time_deriv = params->get<int>("Number Of Time Derivatives");
#ifdef ALBANY_DTK
  bool output_dtk_field = params->get<bool>("Output DTK Field to Exodus", false);
  strvec_t sol_dtk_tag_name = {"Exodus Solution DTK Name",
                               "Exodus SolutionDot DTK Name",
                               "Exodus SolutionDotDot DTK Name"};

  strvec_t sol_dtk_id_name = {"solution dtk",
                              "solution_dot dtk",
                              "solution_dotdot dtk"};
#endif
  //IKT FIXME? - currently won't write dxdp to output file if problem is steady,
  //as this output doesn't work in same way.  May want to change in the future.
  const auto& sens_method = params->get<std::string>("Sensitivity Method","None");

  output_sens_field = num_params > 0 && num_time_deriv > 0 && sens_method != "None";

  //Create tag and id arrays for sensitivity field (dxdp or dgdp)
  strvec_t sol_sens_tag_name_vec;
  strvec_t sol_sens_id_name_vec;
  if (sens_method == "Forward") {
    sol_sens_tag_name_vec.resize(num_params);
    sol_sens_id_name_vec.resize(num_params);
    for (int np=0; np<num_params; np++) {
      sol_sens_tag_name_vec[np] = "Exodus Solution Sensitivity Name" + std::to_string(np);
      sol_sens_id_name_vec[np] = "sensitivity dxdp" + std::to_string(np);
    }
  } else if (sens_method == "Adjoint") {
    //Adjoint sensitivities can only be computed for 1 response/parameter at a time.
    sol_sens_tag_name_vec.resize(1);
    sol_sens_id_name_vec.resize(1);
    sol_sens_tag_name_vec[0] = "Exodus Solution Sensitivity Name";
    //WARNING IKT 8/24/2021: I am not sure that the following will do the right thing in the case the parameter
    //p is not defined on the entire mesh.  A different way of observing dgdp may need to be implemented
    //for that case.  Also note that dgdp will not be written correctly to the mesh for the case of a scalar (vs. distributed) parameter.
    const int resp_fn_index = params->get<int>("Response Function Index");
    const int param_sens_index = params->get<int>("Sensitivity Parameter Index");
    sol_sens_id_name_vec[0] = "sensitivity dg" + std::to_string(resp_fn_index) + "dp" + std::to_string(param_sens_index);
  }

  if (save_solution_field) {
    solution_field.resize(num_time_deriv + 1);
    solution_field_dtk.resize(num_time_deriv + 1);
    solution_field_dxdp.resize(num_params);

    for (int num_vecs = 0; num_vecs <= num_time_deriv; num_vecs++) {
      const auto& name = params->get(sol_tag_name[num_vecs],sol_id_name[num_vecs]);
      solution_field[num_vecs] = add_field_to_mesh<double>(name,true,true,neq);
#if defined(ALBANY_DTK)
      if (output_dtk_field == true) {
        const auto& dtk_name = params->get(sol_dtk_tag_name[num_vecs], sol_dtk_id_name[num_vecs]);
        solution_field_dtk[num_vecs] = add_field_to_mesh<double>(dtk_name,true,true,neq);
      }
#endif
    }
  }

  //Transient sensitivities output to Exodus
  if(output_sens_field) {
    const int num_sens = (sens_method == "Forward") ? num_params : 1;
    for (int np = 0; np < num_sens; np++) {
      const auto sens_name = params->get(sol_sens_tag_name_vec[np], sol_sens_id_name_vec[np]);
      solution_field_dxdp[np] = add_field_to_mesh<double>(sens_name,true,true,neq);

    }
  }
}

void OrdinarySTKFieldContainer::
fillVector(Thyra_Vector&                         field_vector,
           const std::string&                    field_name,
           const Teuchos::RCP<const DOFManager>& field_dof_mgr,
           const bool                            overlapped)
{
  fillVectorImpl(field_vector, field_name, field_dof_mgr, overlapped);
}

void OrdinarySTKFieldContainer::
fillSolnVector(Thyra_Vector&                         solution,
               const Teuchos::RCP<const DOFManager>& solution_dof_mgr,
               const bool overlapped)
{
  fillVectorImpl(solution, solution_field[0]->name(), solution_dof_mgr, overlapped);
}

void OrdinarySTKFieldContainer::
fillSolnMultiVector(Thyra_MultiVector&                    solution,
                    const Teuchos::RCP<const DOFManager>& solution_dof_mgr,
                    const bool                            overlapped)
{
  for (int icomp = 0; icomp < solution.domain()->dim(); ++icomp) {
    auto col = solution.col(icomp);
    TEUCHOS_TEST_FOR_EXCEPTION (col.is_null(), std::runtime_error,
        "Error! Could not extract column from multivector.\n");

    fillVectorImpl(*col, solution_field[icomp]->name(), solution_dof_mgr, overlapped);
  }
}

void OrdinarySTKFieldContainer::
fillSolnSensitivity(Thyra_MultiVector&                    dxdp,
                    const Teuchos::RCP<const DOFManager>& solution_dof_mgr,
                    const bool                            overlapped)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    dxdp.domain()->dim() != num_params, std::runtime_error,
    "Error in fillSolnSensitivity! Wrong number of vectors in dxdp.\n"
    "  - num vectors: " << dxdp.domain()->dim() << "\n"
    "  - num_params : " << num_params << "\n");

  for (int iparam = 0; iparam < solution_field_dxdp.size(); ++iparam) {
    auto col = dxdp.col(iparam);
    TEUCHOS_TEST_FOR_EXCEPTION (col.is_null(), std::runtime_error,
        "Error! Could not extract column from multivector.\n");

    fillVectorImpl(*col, solution_field_dxdp[iparam]->name(), solution_dof_mgr, overlapped);
  }
}

void OrdinarySTKFieldContainer::
saveVector (const Thyra_Vector&  field_vector,
            const std::string&   field_name,
            const dof_mgr_ptr_t& field_dof_mgr,
            const bool           overlapped)
{
  saveVectorImpl (field_vector, field_name, field_dof_mgr, overlapped);
}

void OrdinarySTKFieldContainer::
saveSolnVector (const Thyra_Vector& soln,
                const mv_ptr_t&     soln_dxdp,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  if (save_solution_field) {
    saveVectorImpl (soln, solution_field[0]->name(), sol_dof_mgr, overlapped);
  }

  if (soln_dxdp != Teuchos::null and output_sens_field) {
    TEUCHOS_TEST_FOR_EXCEPTION(
      soln_dxdp->domain()->dim() != num_params, std::runtime_error,
      "Error in saveSolnVector! Wrong number of vectors in soln_dxdp.\n"
      "  - num vectors: " << soln_dxdp->domain()->dim() << "\n"
      "  - num_params : " << num_params << "\n");

    for (int np = 0; np < num_params; np++) {
      saveVectorImpl (*soln_dxdp->col(np), solution_field_dxdp[np]->name(), sol_dof_mgr, overlapped);
    }
  }
}

void OrdinarySTKFieldContainer::
saveSolnVector (const Thyra_Vector&  soln,
                const mv_ptr_t&      soln_dxdp,
                const Thyra_Vector&  soln_dot,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  saveSolnVector(soln,soln_dxdp, sol_dof_mgr, overlapped);
  if (save_solution_field) {
    saveVectorImpl (soln_dot, solution_field[1]->name(), sol_dof_mgr, overlapped);
  }
}

void OrdinarySTKFieldContainer::
saveSolnVector (const Thyra_Vector&  soln,
                const mv_ptr_t&      soln_dxdp,
                const Thyra_Vector&  soln_dot,
                const Thyra_Vector&  soln_dotdot,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  saveSolnVector(soln, soln_dxdp, soln_dot, sol_dof_mgr, overlapped);
  if (save_solution_field) {
    saveVectorImpl (soln_dotdot, solution_field[2]->name(), sol_dof_mgr, overlapped);
  }
}

void OrdinarySTKFieldContainer::
saveSolnMultiVector (const Thyra_MultiVector& soln,
                     const mv_ptr_t&          soln_dxdp,
                     const dof_mgr_ptr_t&     sol_dof_mgr,
                     const bool               overlapped)
{
  for (int icomp = 0; icomp < soln.domain()->dim(); ++icomp) {
    saveVectorImpl (*soln.col(icomp), solution_field[icomp]->name(), sol_dof_mgr, overlapped);
  }

  if (soln_dxdp != Teuchos::null and output_sens_field) {
    TEUCHOS_TEST_FOR_EXCEPTION(
      soln_dxdp->domain()->dim() != num_params, std::runtime_error,
      "Error in saveSolnVector! Wrong number of vectors in soln_dxdp.\n"
      "  - num vectors: " << soln_dxdp->domain()->dim() << "\n"
      "  - num_params : " << num_params << "\n");

    for (int np = 0; np < num_params; np++) {
      saveVectorImpl (*soln_dxdp->col(np), solution_field_dxdp[np]->name(), sol_dof_mgr, overlapped);
    }
  }
}

void OrdinarySTKFieldContainer::
saveResVector (const Thyra_Vector&  res,
               const dof_mgr_ptr_t& dof_mgr,
               const bool           overlapped)
{
  saveVectorImpl (res, residual_field->name(), dof_mgr, overlapped);
}

void OrdinarySTKFieldContainer::
transferSolutionToCoords()
{
  TEUCHOS_TEST_FOR_EXCEPTION (!this->solutionFieldContainer, std::logic_error,
    "Error OrdinarySTKFieldContainer::transferSolutionToCoords not called from a solution field container.\n");

  STKFieldContainerHelper::copySTKField(*solution_field[0], *this->coordinates_field);
}

void OrdinarySTKFieldContainer::
fillVectorImpl(Thyra_Vector&                         field_vector,
               const std::string&                    field_name,
               const Teuchos::RCP<const DOFManager>& field_dof_mgr,
               const bool                            overlapped)
{
  // Figure out if it's a nodal or elem field
  const auto& fp = field_dof_mgr->getGeometricFieldPattern();
  const auto& ftopo = field_dof_mgr->get_topology();
  std::vector<int> entity_dims_with_dofs;
  for (unsigned dim=0; dim<=ftopo.getDimension(); ++dim) {
    if (fp->getSubcellIndices(dim,0).size()>0) {
      entity_dims_with_dofs.push_back(dim);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION (entity_dims_with_dofs.size()!=1, std::runtime_error,
      "Error! We cannot save fields that are defined on n!=1 type of entity.\n"
      "  - field name: " << field_name << "\n"
      "  - entities with dofs: " << util::join(entity_dims_with_dofs,",") << "\n");

  auto field_entity_rank = static_cast<stk::topology::rank_t>(entity_dims_with_dofs[0]);

  auto* raw_field = metaData->get_field(field_entity_rank, field_name);
  ALBANY_EXPECT (raw_field != nullptr,
      "Error! Something went wrong while retrieving a field.\n");

  const auto* field = this->metaData->template get_field<double>(field_entity_rank, field_name);
  STKFieldContainerHelper::fillVector(field_vector, *field, *this->bulkData, field_dof_mgr, overlapped);
}

void OrdinarySTKFieldContainer::
saveVectorImpl (const Thyra_Vector&  field_vector,
                const std::string&   field_name,
                const dof_mgr_ptr_t& field_dof_mgr,
                const bool           overlapped)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!this->solutionFieldContainer, std::logic_error,
    "Error OrdinarySTKFieldContainer::saveVectorImpl not called from a solution field container.\n");

  // Figure out if it's a nodal or elem field
  const auto& fp = field_dof_mgr->getGeometricFieldPattern();
  const auto& ftopo = field_dof_mgr->get_topology();
  std::vector<int> entity_dims_with_dofs;
  for (unsigned dim=0; dim<=ftopo.getDimension(); ++dim) {
    if (fp->getSubcellIndices(dim,0).size()>0) {
      entity_dims_with_dofs.push_back(dim);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION (entity_dims_with_dofs.size()!=1, std::runtime_error,
      "Error! We cannot save fields that are defined on n!=1 type of entity.\n"
      "  - field name: " << field_name << "\n"
      "  - entities with dofs: " << util::join(entity_dims_with_dofs,",") << "\n");

  auto field_entity_rank = static_cast<stk::topology::rank_t>(entity_dims_with_dofs[0]);

  auto* raw_field = metaData->get_field(field_entity_rank, field_name);
  ALBANY_EXPECT (raw_field != nullptr,
      "Error! Something went wrong while retrieving a field.\n");

  auto* field = this->metaData->template get_field<double>(field_entity_rank, field_name);
  STKFieldContainerHelper::saveVector(field_vector, *field, *this->bulkData, field_dof_mgr, overlapped);
}

}  // namespace Albany
