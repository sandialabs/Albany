//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_DISCRETIZATION_HPP
#define ALBANY_ABSTRACT_DISCRETIZATION_HPP

#include "Albany_config.h"

#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "Albany_NodalDOFManager.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Shards_Array.hpp"
#include "Shards_CellTopologyData.h"

#include "Albany_ThyraTypes.hpp"

namespace Albany {

class AbstractDiscretization
{
 public:
  typedef std::map<std::string, Teuchos::RCP<AbstractDiscretization>>
      SideSetDiscretizationsType;

  //! Constructor
  AbstractDiscretization() = default;

  //! Prohibit copying
  AbstractDiscretization(const AbstractDiscretization&) = delete;
  AbstractDiscretization&
  operator=(const AbstractDiscretization&) = default;

  //! Destructor
  virtual ~AbstractDiscretization() = default;

  //! Get node vector space (owned and overlapped)
  virtual Teuchos::RCP<const Thyra_VectorSpace>
  getNodeVectorSpace() const = 0;
  virtual Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapNodeVectorSpace() const = 0;

  //! Get solution DOF vector space (owned and overlapped).
  virtual Teuchos::RCP<const Thyra_VectorSpace>
  getVectorSpace() const = 0;
  virtual Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapVectorSpace() const = 0;

  //! Get Field node vector space (owned and overlapped)
  virtual Teuchos::RCP<const Thyra_VectorSpace>
  getNodeVectorSpace(const std::string& field_name) const = 0;
  virtual Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapNodeVectorSpace(const std::string& field_name) const = 0;

  //! Get Field vector space (owned and overlapped)
  virtual Teuchos::RCP<const Thyra_VectorSpace>
  getVectorSpace(const std::string& field_name) const = 0;
  virtual Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapVectorSpace(const std::string& field_name) const = 0;

  //! Create a Jacobian operator (owned and overlapped)
  virtual Teuchos::RCP<Thyra_LinearOp>
  createJacobianOp() const = 0;
  virtual Teuchos::RCP<Thyra_LinearOp>
  createOverlapJacobianOp() const = 0;

  //! Returns boolean telling code whether explicit scheme is used (needed for
  //! Aeras problems only)
  virtual bool
  isExplicitScheme() const = 0;

  //! Get Node set lists
  virtual const NodeSetList&
  getNodeSets() const = 0;
  virtual const NodeSetGIDsList&
  getNodeSetGIDs() const = 0;
  virtual const NodeSetCoordList&
  getNodeSetCoords() const = 0;

  //! Get Side set lists
  virtual const SideSetList&
  getSideSets(const int ws) const = 0;

  //! Get map from (Ws, El, Local Node, Eq) -> unkLID
  virtual const Conn&
  getWsElNodeEqID() const = 0;

  //! Get map from (Ws, El, Local Node) -> unkGID
  virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>::type&
  getWsElNodeID() const = 0;

  //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for
  //! both scalar and vector fields
  virtual const std::vector<IDArray>&
  getElNodeEqID(const std::string& field_name) const = 0;

  //! Get Dof Manager of field field_name
  virtual const NodalDOFManager&
  getDOFManager(const std::string& field_name) const = 0;

  //! Get Dof Manager of field field_name
  virtual const NodalDOFManager&
  getOverlapDOFManager(const std::string& field_name) const = 0;

  //! Retrieve coodinate ptr_field (ws, el, node)
  virtual const WorksetArray<
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>::type&
  getCoords() const = 0;

  //! Get coordinates (overlap map).
  virtual const Teuchos::ArrayRCP<double>&
  getCoordinates() const = 0;
  //! Set coordinates (overlap map) for mesh adaptation.
  virtual void
  setCoordinates(const Teuchos::ArrayRCP<const double>& c) = 0;

  //! Print the coords for mesh debugging
  virtual void
  printCoords() const = 0;

  //! Get sideSet discretizations map
  virtual const SideSetDiscretizationsType&
  getSideSetDiscretizations() const = 0;

  //! Get the map side_id->side_set_elem_id
  virtual const std::map<std::string, std::map<GO, GO>>&
  getSideToSideSetCellMap() const = 0;

  //! Get the map side_node_id->side_set_cell_node_id
  virtual const std::map<std::string, std::map<GO, std::vector<int>>>&
  getSideNodeNumerationMap() const = 0;

  //! Get MeshStruct
  virtual Teuchos::RCP<AbstractMeshStruct>
  getMeshStruct() const = 0;

  //! Set stateArrays
  virtual void
  setStateArrays(StateArrays& sa) = 0;

  //! Get stateArrays
  virtual StateArrays&
  getStateArrays() = 0;

  //! Get nodal parameters state info struct
  virtual const StateInfoStruct&
  getNodalParameterSIS() const = 0;

  //! Retrieve Vector (length num worksets) of element block names
  virtual const WorksetArray<std::string>::type&
  getWsEBNames() const = 0;

  //! Retrieve Vector (length num worksets) of Physics Index
  virtual const WorksetArray<int>::type&
  getWsPhysIndex() const = 0;

  //! Retrieve connectivity map from elementGID to workset
  virtual WsLIDList&
  getElemGIDws() = 0;
  virtual const WsLIDList&
  getElemGIDws() const = 0;

  //! Flag if solution has a restart values -- used in Init Cond
  virtual bool
  hasRestartSolution() const = 0;

  //! File time of restart solution
  virtual double
  restartDataTime() const = 0;

  //! Get number of spatial dimensions
  virtual int
  getNumDim() const = 0;

  //! Get number of total DOFs per node
  virtual int
  getNumEq() const = 0;

  //! Get Numbering for layered mesh (mesh structred in one direction)
  virtual Teuchos::RCP<LayeredMeshNumbering<LO>>
  getLayeredMeshNumbering() const = 0;

  // --- Get/set solution/residual/field vectors to/from mesh --- //
  virtual Teuchos::RCP<Thyra_Vector>
  getSolutionField(bool overlapped = false) const = 0;

  virtual Teuchos::RCP<Thyra_MultiVector>
  getSolutionMV(bool overlapped = false) const = 0;

  virtual void
  getField(Thyra_Vector& field_vector, const std::string& field_name) const = 0;
  virtual void
  setField(
      const Thyra_Vector& field_vector,
      const std::string&  field_name,
      bool                overlapped) = 0;

  // --- Methods to write solution in the output file --- //

  //! Write the solution to the output file. Calls next two together.
  virtual void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double        time,
      const bool          overlapped = false) = 0;
  virtual void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const double        time,
      const bool          overlapped = false) = 0;
  virtual void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const double        time,
      const bool          overlapped = false) = 0;
  virtual void
  writeSolutionMV(
      const Thyra_MultiVector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double             time,
      const bool               overlapped = false) = 0;
  //! Write the solution to the mesh database.
  virtual void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double        time,
      const bool          overlapped = false) = 0;
  virtual void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const double        time,
      const bool          overlapped = false) = 0;
  virtual void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const double        time,
      const bool          overlapped = false) = 0;
  virtual void
  writeSolutionMVToMeshDatabase(
      const Thyra_MultiVector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double             time,
      const bool               overlapped = false) = 0;

  //! Write the solution to file. Must call writeSolution first.
  virtual void
  writeSolutionToFile(
      const Thyra_Vector& solution,
      const double        time,
      const bool          overlapped = false) = 0;
  virtual void
  writeSolutionMVToFile(
      const Thyra_MultiVector& solution,
      const double             time,
      const bool               overlapped = false) = 0;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_DISCRETIZATION_HPP
