//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP
#define ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP

#include "Albany_AbstractMeshFieldAccessor.hpp"

#include <stk_mesh/base/Field.hpp>
#include "Teuchos_RCP.hpp"

namespace Albany {

/*!
 * \brief Abstract interface for an STK field container
 *
 */
class AbstractSTKFieldContainer : public AbstractMeshFieldAccessor
{
public:
  // STK field (scalar/vector/tensor , Node/Cell)
  using STKFieldType = stk::mesh::Field<double>;
  // STK int field
  using STKIntState  = stk::mesh::Field<int>;

  using STKState   = std::vector<STKFieldType*>;

  AbstractSTKFieldContainer (bool solutionFieldContainer_) : solutionFieldContainer(solutionFieldContainer_) {}

  //! Destructor
  virtual ~AbstractSTKFieldContainer() = default;

  // Coordinates field ALWAYS in 3D
  const STKFieldType*
  getCoordinatesField3d() const
  {
    return coordinates_field3d;
  }
  STKFieldType*
  getCoordinatesField3d()
  {
    return coordinates_field3d;
  }

  const STKFieldType*
  getCoordinatesField() const
  {
    return coordinates_field;
  }
  STKFieldType*
  getCoordinatesField()
  {
    return coordinates_field;
  }

  STKIntState*
  getProcRankField()
  {
    return proc_rank_field;
  }

  STKState&
  getCellScalarStates()
  {
    return cell_scalar_states;
  }
  STKState&
  getCellVectorStates()
  {
    return cell_vector_states;
  }
  STKState&
  getCellTensorStates()
  {
    return cell_tensor_states;
  }
  STKState&
  getQPScalarStates()
  {
    return qpscalar_states;
  }
  STKState&
  getQPVectorStates()
  {
    return qpvector_states;
  }
  STKState&
  getQPTensorStates()
  {
    return qptensor_states;
  }

  virtual void transferSolutionToCoords() = 0;

protected:
  // Note: for 3d meshes, coordinates_field3d==coordinates_field (they point to
  // the same field).
  //       Otherwise, coordinates_field3d stores coordinates in 3d (useful for
  //       non-flat 2d meshes)
  STKFieldType*    coordinates_field3d = nullptr;
  STKFieldType*    coordinates_field   = nullptr;
  STKIntState*     proc_rank_field     = nullptr;

  STKState                  cell_scalar_states;
  STKState                  cell_vector_states;
  STKState                  cell_tensor_states;
  STKState                  qpscalar_states;
  STKState                  qpvector_states;
  STKState                  qptensor_states;

  const bool solutionFieldContainer;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP
