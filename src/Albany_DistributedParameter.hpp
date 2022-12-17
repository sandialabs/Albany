//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DISTRIBUTED_PARAMETER_HPP
#define ALBANY_DISTRIBUTED_PARAMETER_HPP

#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_DOFManager.hpp"
#include "Albany_StateInfoStruct.hpp" // For IDArray
#include "Albany_ThyraTypes.hpp"
#include "Albany_CombineAndScatterManager.hpp"

namespace Albany {

//! Class for storing distributed parameters
class DistributedParameter {
public:

  using dof_mgr_ptr_t = Teuchos::RCP<const DOFManager>;

  //! Constructor(s)

  // Parameter defined on all the elements of the dof mgr
  DistributedParameter(const std::string& param_name_,
                       const dof_mgr_ptr_t& dof_mgr)
   : param_name(param_name_)
   , m_dof_mgr (dof_mgr)
  {
    // Sanity checks
    TEUCHOS_TEST_FOR_EXCEPTION(m_dof_mgr.is_null(), std::runtime_error,
        "Error! Input dof manager is null.\n");

    auto owned_vs = m_dof_mgr->vs();
    auto overlapped_vs = m_dof_mgr->ov_vs();

    owned_vec = Thyra::createMember(owned_vs);
    overlapped_vec = Thyra::createMember(overlapped_vs);

    lower_bounds_vec = Thyra::createMember(owned_vs);
    upper_bounds_vec = Thyra::createMember(owned_vs);

    cas_manager = createCombineAndScatterManager(owned_vs, overlapped_vs);
  }

  //! Destructor
  virtual ~DistributedParameter() {}

  //! Get name
  const std::string& name() const { return param_name; }

  //! Get vector space 
  Teuchos::RCP<const Thyra_VectorSpace> vector_space() const { return owned_vec->space(); }

  //! Get overlap vector space
  Teuchos::RCP<const Thyra_VectorSpace> overlap_vector_space() const { return overlapped_vec->space(); }

  //! Get vector
  Teuchos::RCP<Thyra_Vector> vector() const { return owned_vec; }

  //! Get overlapped vector
  Teuchos::RCP<Thyra_Vector> overlapped_vector() const { return overlapped_vec; }

  //! Get lower bounds vector
  Teuchos::RCP<Thyra_Vector> lower_bounds_vector() const { return lower_bounds_vec; }

  //! Get upper bounds vector
  Teuchos::RCP<Thyra_Vector> upper_bounds_vector() const { return upper_bounds_vec; }

  //! Fill overlapped vector from owned vector (CombineMode = INSERT)
  void scatter() const {
    cas_manager->scatter(owned_vec, overlapped_vec, CombineMode::INSERT);
  }

  //! Fill owned vector from overlapped vector (CombineMode = ZERO)
  void combine() const {
    // Note: this allows one to fill the overlapped vector (rather than the owned)
    //       during the evaluation phase, and simply copy what's local in the
    //       overlapped_vec into the owned_vec
    cas_manager->combine(overlapped_vec, owned_vec, CombineMode::ZERO);
  }

  dof_mgr_ptr_t get_dof_mgr () const { return m_dof_mgr; }

  //! Get the CombineAndScatterManager for this parameter
  Teuchos::RCP<const CombineAndScatterManager> get_cas_manager () const { return cas_manager; }

protected:

  // Set the m_elem_dof_lids dual view to -1 outside the mesh part,
  // and builds vectors/vectorSpaces restricted to mesh_part
  void compute_elem_dof_lids(const std::string& mesh_part);

  //! Name of parameter
  std::string param_name;

  //! Owned and repeated vectors
  Teuchos::RCP<Thyra_Vector>        owned_vec;
  Teuchos::RCP<Thyra_Vector>        overlapped_vec;

  //! Lower and upper bounds (they are null if never provided)
  Teuchos::RCP<Thyra_Vector>        lower_bounds_vec;
  Teuchos::RCP<Thyra_Vector>        upper_bounds_vec;

  //! The manager for scatter/combine operation
  Teuchos::RCP<const CombineAndScatterManager> cas_manager;

  // The DOFManager for this parameter
  dof_mgr_ptr_t m_dof_mgr;
};

} // namespace Albany

#endif // ALBANY_DISTRIBUTED_PARAMETER_HPP
