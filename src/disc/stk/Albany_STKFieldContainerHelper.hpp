//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STK_FIELD_CONTAINER_HELPER_HPP
#define ALBANY_STK_FIELD_CONTAINER_HELPER_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_DOFManager.hpp"

#include <stk_mesh/base/Bucket.hpp>

namespace Albany {

class GlobalLocalIndexer;

// FieldType can be either scalar or vector, the code is the same.
struct STKFieldContainerHelper
{
  using FieldType = stk::mesh::Field<double>;

  // Fill (aka get) and save (aka set) methods
  // If passing no components, we do all fields in the dof mgr

  static void fillVector (      Thyra_Vector& field_thyra,
                          const FieldType& field_stk,
                          const stk::mesh::BulkData& bulkData,
                          const Teuchos::RCP<const DOFManager>& dof_mgr,
                          const bool overlapped);
  static void fillVector (      Thyra_Vector& field_thyra,
                          const FieldType& field_stk,
                          const stk::mesh::BulkData& bulkData,
                          const Teuchos::RCP<const DOFManager>& dof_mgr,
                          const bool overlapped,
                          const std::vector<int>& components);

  static void saveVector (const Thyra_Vector& field_thyra,
                                FieldType& field_stk,
                          const stk::mesh::BulkData& bulkData,
                          const Teuchos::RCP<const DOFManager>& dof_mgr,
                          const bool overlapped);
  static void saveVector (const Thyra_Vector& field_thyra,
                                FieldType& field_stk,
                          const stk::mesh::BulkData& bulkData,
                          const Teuchos::RCP<const DOFManager>& dof_mgr,
                          const bool overlapped,
                          const std::vector<int>& components);

  // Convenience function to copy one field's contents to another
  static void copySTKField(const FieldType& source, FieldType& target);
};

} // namespace Albany

#endif // ALBANY_STK_FIELD_CONTAINER_HELPER_HPP
