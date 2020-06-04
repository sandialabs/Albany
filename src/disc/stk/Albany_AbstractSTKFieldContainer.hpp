//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP
#define ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP

#include "Albany_config.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

// This include is added in Tpetra branch to get all the necessary
// Tpetra includes (e.g., Tpetra_Vector.hpp, Tpetra_Map.hpp, etc.)
#include "Albany_DataTypes.hpp"

#include "Albany_AbstractFieldContainer.hpp"
#include "Albany_NodalDOFManager.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_Utils.hpp"

#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>

namespace Albany {

/*!
 * \brief Abstract interface for an STK field container
 *
 */
class AbstractSTKFieldContainer : public AbstractFieldContainer
{
 public:
  // Tensor per Node/Cell  - (Node, Dim, Dim) or (Cell,Dim,Dim)
  typedef stk::mesh::Field<double, stk::mesh::Cartesian, stk::mesh::Cartesian>
      TensorFieldType;
  // Vector per Node/Cell  - (Node, Dim) or (Cell,Dim)
  typedef stk::mesh::Field<double, stk::mesh::Cartesian> VectorFieldType;
  // Scalar per Node/Cell  - (Node) or (Cell)
  typedef stk::mesh::Field<double> ScalarFieldType;
  // One int scalar per Node/Cell  - (Node) or (Cell)
  typedef stk::mesh::Field<int> IntScalarFieldType;
  // int vector per Node/Cell  - (Node,Dim/VecDim) or (Cell,Dim/VecDim)
  typedef stk::mesh::Field<int, stk::mesh::Cartesian> IntVectorFieldType;

  typedef stk::mesh::Cartesian QPTag;  // need to invent shards::ArrayDimTag
  // Tensor per QP   - (Cell, QP, Dim, Dim)
  typedef stk::mesh::
      Field<double, QPTag, stk::mesh::Cartesian, stk::mesh::Cartesian>
          QPTensorFieldType;
  // Vector per QP   - (Cell, QP, Dim)
  typedef stk::mesh::Field<double, QPTag, stk::mesh::Cartesian>
      QPVectorFieldType;
  // One scalar per QP   - (Cell, QP)
  typedef stk::mesh::Field<double, QPTag> QPScalarFieldType;

  typedef std::vector<const std::string*> ScalarValueState;
  typedef std::vector<QPScalarFieldType*> QPScalarState;
  typedef std::vector<QPVectorFieldType*> QPVectorState;
  typedef std::vector<QPTensorFieldType*> QPTensorState;

  typedef std::vector<ScalarFieldType*> ScalarState;
  typedef std::vector<VectorFieldType*> VectorState;
  typedef std::vector<TensorFieldType*> TensorState;

  typedef std::map<std::string, double>              MeshScalarState;
  typedef std::map<std::string, std::vector<double>> MeshVectorState;

  typedef std::map<std::string, int>              MeshScalarIntegerState;
  typedef std::map<std::string, std::vector<int>> MeshVectorIntegerState;
  //! Destructor
  virtual ~AbstractSTKFieldContainer(){};

  virtual void
  addStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis) = 0;

  // Coordinates field ALWAYS in 3D
  const VectorFieldType*
  getCoordinatesField3d() const
  {
    return coordinates_field3d;
  }
  VectorFieldType*
  getCoordinatesField3d()
  {
    return coordinates_field3d;
  }

  const VectorFieldType*
  getCoordinatesField() const
  {
    return coordinates_field;
  }
  VectorFieldType*
  getCoordinatesField()
  {
    return coordinates_field;
  }
  IntScalarFieldType*
  getProcRankField()
  {
    return proc_rank_field;
  }

  ScalarValueState&
  getScalarValueStates()
  {
    return scalarValue_states;
  }
  MeshScalarState&
  getMeshScalarStates()
  {
    return mesh_scalar_states;
  }
  MeshVectorState&
  getMeshVectorStates()
  {
    return mesh_vector_states;
  }
  MeshScalarIntegerState&
  getMeshScalarIntegerStates()
  {
    return mesh_scalar_integer_states;
  }
  MeshVectorIntegerState&
  getMeshVectorIntegerStates()
  {
    return mesh_vector_integer_states;
  }
  ScalarState&
  getCellScalarStates()
  {
    return cell_scalar_states;
  }
  VectorState&
  getCellVectorStates()
  {
    return cell_vector_states;
  }
  TensorState&
  getCellTensorStates()
  {
    return cell_tensor_states;
  }
  QPScalarState&
  getQPScalarStates()
  {
    return qpscalar_states;
  }
  QPVectorState&
  getQPVectorStates()
  {
    return qpvector_states;
  }
  QPTensorState&
  getQPTensorStates()
  {
    return qptensor_states;
  }
  const StateInfoStruct&
  getNodalSIS() const
  {
    return nodal_sis;
  }
  const StateInfoStruct&
  getNodalParameterSIS() const
  {
    return nodal_parameter_sis;
  }

  std::map<std::string, double>&
  getTime()
  {
    return time;
  }

  virtual void
  fillSolnVector(
      Thyra_Vector&                                soln,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  fillVector(
      Thyra_Vector&                                field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager) = 0;
  virtual void
  fillSolnMultiVector(
      Thyra_MultiVector&                           soln,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  saveVector(
      const Thyra_Vector&                          field_vector,
      const std::string&                           field_name,
      stk::mesh::Selector&                         field_selection,
      const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
      const NodalDOFManager&                       nodalDofManager) = 0;
  virtual void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      const Thyra_Vector&                          soln_dot,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  saveSolnVector(
      const Thyra_Vector&                          soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      const Thyra_Vector&                          soln_dot,
      const Thyra_Vector&                          soln_dotdot,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  saveResVector(
      const Thyra_Vector&                          res,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;
  virtual void
  saveSolnMultiVector(
      const Thyra_MultiVector&                     soln,
      const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
      stk::mesh::Selector&                         sel,
      const Teuchos::RCP<const Thyra_VectorSpace>& node_vs) = 0;

  virtual void
  transferSolutionToCoords() = 0;

 protected:
  // Note: for 3d meshes, coordinates_field3d==coordinates_field (they point to
  // the same field).
  //       Otherwise, coordinates_field3d stores coordinates in 3d (useful for
  //       non-flat 2d meshes)
  VectorFieldType*    coordinates_field3d;
  VectorFieldType*    coordinates_field;
  IntScalarFieldType* proc_rank_field;

  ScalarValueState       scalarValue_states;
  MeshScalarState        mesh_scalar_states;
  MeshVectorState        mesh_vector_states;
  MeshScalarIntegerState mesh_scalar_integer_states;
  MeshVectorIntegerState mesh_vector_integer_states;
  ScalarState            cell_scalar_states;
  VectorState            cell_vector_states;
  TensorState            cell_tensor_states;
  QPScalarState          qpscalar_states;
  QPVectorState          qpvector_states;
  QPTensorState          qptensor_states;

  StateInfoStruct nodal_sis;
  StateInfoStruct nodal_parameter_sis;

  std::map<std::string, double> time;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP
