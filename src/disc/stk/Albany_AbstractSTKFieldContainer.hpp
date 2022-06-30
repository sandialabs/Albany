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

// // This include is added in Tpetra branch to get all the necessary
// // Tpetra includes (e.g., Tpetra_Vector.hpp, Tpetra_Map.hpp, etc.)
// #include "Albany_DataTypes.hpp"

// #include "Albany_AbstractFieldContainer.hpp"
#include "Albany_DOF.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_Utils.hpp"

#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/Field.hpp>

namespace Albany {

/*!
 * \brief Abstract interface for an STK field container
 *
 */
// class AbstractSTKFieldContainer : public AbstractFieldContainer
class AbstractSTKFieldContainer
{
public:
  // ------------------------- Public Types ------------------------------- //
  using vs_ptr_t  = Teuchos::RCP<const Thyra_VectorSpace>;
  using cmv_ptr_t = Teuchos::RCP<const Thyra_MultiVector>;
  using Cartesian = stk::mesh::Cartesian;
  using QPTag     = stk::mesh::Cartesian;

  template<typename... Args>
  using Field = stk::mesh::Field<Args...>;

  // Tensor per Node/Cell  - (Node, Dim, Dim) or (Cell,Dim,Dim)
  typedef Field<double, Cartesian, Cartesian> TensorFieldType;
  // Vector per Node/Cell  - (Node, Dim) or (Cell,Dim)
  typedef Field<double, Cartesian> VectorFieldType;
  // Scalar per Node/Cell  - (Node) or (Cell)
  typedef Field<double> ScalarFieldType;
  // One int scalar per Node/Cell  - (Node) or (Cell)
  typedef Field<int> IntScalarFieldType;
  // int vector per Node/Cell  - (Node,Dim/VecDim) or (Cell,Dim/VecDim)
  typedef Field<int, Cartesian> IntVectorFieldType;

  // Tensor per QP   - (Cell, QP, Dim, Dim)
  typedef Field<double, QPTag, Cartesian, Cartesian> QPTensorFieldType;
  // Vector per QP   - (Cell, QP, Dim)
  typedef Field<double, QPTag, Cartesian> QPVectorFieldType;

  // One scalar per QP   - (Cell, QP)
  typedef Field<double, QPTag> QPScalarFieldType;

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
  typedef std::map<std::string, GO>               MeshScalarInteger64State;
  typedef std::map<std::string, std::vector<int>> MeshVectorIntegerState;

  // ---------------------------------------------------------------------- //

  //! Constructor and Destructor
  AbstractSTKFieldContainer(bool solutionFieldContainer_) : proc_rank_field(nullptr), solutionFieldContainer(solutionFieldContainer_) {};

  virtual ~AbstractSTKFieldContainer () = default;

  // Add stk fields from state structs data
  virtual void addStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis) = 0;

  // Coordinates field ALWAYS in 3D, and of intrinsic mesh dimension
  const VectorFieldType* getCoordinatesField3d() const { return coordinates_field3d; }
        VectorFieldType* getCoordinatesField3d()       { return coordinates_field3d; }
  const VectorFieldType* getCoordinatesField()   const { return coordinates_field;   }
        VectorFieldType* getCoordinatesField()         { return coordinates_field;   }

  IntScalarFieldType* getProcRankField() const { return proc_rank_field; }

  // Get states, based on rank, location, data type
  ScalarValueState&           getScalarValueStates()         { return scalarValue_states; }
  MeshScalarState&            getMeshScalarStates()          { return mesh_scalar_states; }
  MeshVectorState&            getMeshVectorStates()          { return mesh_vector_states; }
  MeshScalarIntegerState&     getMeshScalarIntegerStates()   { return mesh_scalar_integer_states; }
  MeshScalarInteger64State&   getMeshScalarInteger64States() { return mesh_scalar_integer_64_states; }
  MeshVectorIntegerState&     getMeshVectorIntegerStates()   { return mesh_vector_integer_states; }
  ScalarState&                getCellScalarStates()          { return cell_scalar_states; }
  VectorState&                getCellVectorStates()          { return cell_vector_states; }
  TensorState&                getCellTensorStates()          { return cell_tensor_states; }
  QPScalarState&              getQPScalarStates()            { return qpscalar_states; }
  QPVectorState&              getQPVectorStates()            { return qpvector_states; }
  QPTensorState&              getQPTensorStates()            { return qptensor_states; }

  const StateInfoStruct& getNodalSIS()          const { return nodal_sis; }
  const StateInfoStruct& getNodalParameterSIS() const { return nodal_parameter_sis; }

  std::map<std::string, double>& getTime() { return time; }

  // Copy fields from STK mesh into Thyra vectors
  // virtual void fillSolnVector (Thyra_Vector&        soln,
  //                             stk::mesh::Selector&  sel,
  //                             const vs_ptr_t&       node_vs) = 0;
  virtual void fillSolnVector (Thyra_Vector& solution,
                               const DOF&    solution_dof) = 0;

  // virtual void fillVector (Thyra_Vector&          field_vector,
  //                          const std::string&     field_name,
  //                          stk::mesh::Selector&   field_selection,
  //                          const vs_ptr_t&        field_node_vs,
  //                          const NodalDOFManager& nodalDofManager) = 0;
  virtual void fillVector (Thyra_Vector&      field_vector,
                           const std::string& field_name,
                           const DOF&         field_dof) = 0;

  // virtual void fillSolnMultiVector (Thyra_MultiVector&    soln,
  //                                   stk::mesh::Selector&  sel,
  //                                   const vs_ptr_t&       node_vs) = 0;
  virtual void fillSolnMultiVector (Thyra_MultiVector& solution,
                                    const DOF&         solution_dof) = 0;

  // Copy fields from Thyra vectors into STK mesh
  // virtual void saveVector (const Thyra_Vector&     field_vector,
  //                          const std::string&      field_name,
  //                          stk::mesh::Selector&    field_selection,
  //                          const vs_ptr_t&         field_node_vs,
  //                          const NodalDOFManager&  nodalDofManager) = 0;
  virtual void saveVector (const Thyra_Vector&  field_vector,
                           const std::string&   field_name,
                           const DOF&           field_dof) = 0;

  // virtual void saveSolnVector (const Thyra_Vector&  soln,
  //                              const cmv_ptr_t&     soln_dxdp,
  //                              stk::mesh::Selector& sel,
  //                              const vs_ptr_t&      node_vs) = 0;
  virtual void saveSolnVector (const Thyra_Vector&  solution,
                               const cmv_ptr_t&     solution_dxdp,
                               const DOF&           solution_dof) = 0;

  // virtual void saveSolnVector (const Thyra_Vector&  soln,
  //                              const cmv_ptr_t&     soln_dxdp,
  //                              const Thyra_Vector&  soln_dot,
  //                              stk::mesh::Selector& sel,
  //                              const vs_ptr_t&      node_vs) = 0;
  virtual void saveSolnVector (const Thyra_Vector&  solution,
                               const cmv_ptr_t&     solution_dxdp,
                               const Thyra_Vector&  solution_dot,
                               const DOF&           solution_dof) = 0;

  // virtual void saveSolnVector (const Thyra_Vector&  soln,
  //                              const cmv_ptr_t&     soln_dxdp,
  //                              const Thyra_Vector&  soln_dot,
  //                              const Thyra_Vector&  soln_dotdot,
  //                              stk::mesh::Selector& sel,
  //                              const vs_ptr_t&      node_vs) = 0;
  virtual void saveSolnVector (const Thyra_Vector&  solution,
                               const cmv_ptr_t&     solution_dxdp,
                               const Thyra_Vector&  solution_dot,
                               const Thyra_Vector&  solution_dotdot,
                               const DOF&           solution_dof) = 0;

  // virtual void saveResVector (const Thyra_Vector&   res,
  //                             stk::mesh::Selector&  sel,
  //                             const vs_ptr_t&       node_vs) = 0;
  virtual void saveResVector (const Thyra_Vector& residual,
                              const DOF&          solution_dof) = 0;

  // virtual void saveSolnMultiVector (const Thyra_MultiVector&  soln,
  //                                   const cmv_ptr_t&          soln_dxdp,
  //                                   stk::mesh::Selector&      sel,
  //                                   const vs_ptr_t&           node_vs) = 0;
  virtual void saveSolnMultiVector (const Thyra_MultiVector&  solution,
                                    const cmv_ptr_t&          solution_dxdp,
                                    const DOF&                solution_dof) = 0;

  virtual void transferSolutionToCoords() = 0;

 protected:
  // Note: for 3d meshes, coordinates_field3d==coordinates_field (they point to
  // the same field).
  //       Otherwise, coordinates_field3d stores coordinates in 3d (useful for
  //       non-flat 2d meshes)
  VectorFieldType*    coordinates_field3d;
  VectorFieldType*    coordinates_field;
  IntScalarFieldType* proc_rank_field;

  ScalarValueState          scalarValue_states;
  MeshScalarState           mesh_scalar_states;
  MeshVectorState           mesh_vector_states;
  MeshScalarIntegerState    mesh_scalar_integer_states;
  MeshScalarInteger64State  mesh_scalar_integer_64_states;
  MeshVectorIntegerState    mesh_vector_integer_states;
  ScalarState               cell_scalar_states;
  VectorState               cell_vector_states;
  TensorState               cell_tensor_states;
  QPScalarState             qpscalar_states;
  QPVectorState             qpvector_states;
  QPTensorState             qptensor_states;

  StateInfoStruct nodal_sis;
  StateInfoStruct nodal_parameter_sis;

  std::map<std::string, double> time;

  const bool solutionFieldContainer;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_FIELD_CONTAINER_HPP
