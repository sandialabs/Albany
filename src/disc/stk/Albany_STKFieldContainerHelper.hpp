//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STK_FIELD_CONTAINER_HELPER_HPP
#define ALBANY_STK_FIELD_CONTAINER_HELPER_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_DOF.hpp"

#include <stk_mesh/base/Bucket.hpp>

namespace Albany {

template<class FieldType>
struct STKFieldContainerHelper
{
  // FieldType can be either scalar or vector, the code is the same.

  // Fill (from stk to thyra) and save (from thyra to stk) routines
  // Parameters:
  //  - vector: the thyra vector where to read/write data from/to
  //  - field_stk: the STK field where to write/read data to/from
  //  - bucket: current node bucket being processed
  //  - vector_dof: the Albany::DOF structure corresponding to the thyra vector
  //  - offset: if reading/writing only a subfield of vector, this is
  //            the offset to the first entry of the subfield (recall that
  //            we always use interleaved ordering of dofs)
  // Note on the field dimensions:
  //  - the thyra vector stores N components, with N the return value
  //    of vector_dof.dof_mgr->getNumFields()
  //  - the stk field has M scalars per node (M=1 for scalar fields)
  //    M can be computed via stk::mesh::field_scalars_per_entity(field_stk,bucket)
  // All of this assumes that N>=M, and the thyra vector might be a
  // "packing" of K separate fields, whose individual dimensions add up to N.

  static void fillVector (      Thyra_Vector& vector,
                          const FieldType& field_stk,
                          const stk::mesh::Bucket& bucket,
                          const DOF& vector_dof,
                          const int offset);

  static void saveVector (const Thyra_Vector& field_thyra,
                                FieldType& field_stk,
                          const stk::mesh::Bucket& bucket,
                          const DOF& field_dof,
                          const int offset);

  // Convenience function to copy one field's contents to another
  static void copySTKField(const FieldType& source, FieldType& target);
};

} // namespace Albany

#endif // ALBANY_STK_FIELD_CONTAINER_HELPER_HPP
