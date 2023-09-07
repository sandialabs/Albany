//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_OMEGAH_DISCRETIZATION_HPP
#define ALBANY_OMEGAH_DISCRETIZATION_HPP

#include "Albany_AbstractDiscretization.hpp"

#include "Albany_OmegahAbstractMesh.hpp"
#include "Albany_NullSpaceUtils.hpp"

namespace Albany {

class OmegahDiscretization : public AbstractDiscretization
{
public:
  OmegahDiscretization(
      const Teuchos::RCP<Teuchos::ParameterList>& discParams,
      const int neq,
      Teuchos::RCP<OmegahAbstractMesh>&           mesh,
      const Teuchos::RCP<const Teuchos_Comm>&     comm,
      const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null,
      const std::map<int, std::vector<std::string>>& sideSetEquations =
          std::map<int, std::vector<std::string>>());

  //! Destructor
  ~OmegahDiscretization() = default;

  void updateMesh ();

  Teuchos::RCP<Thyra_LinearOp>
  createJacobianOp() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get Node set lists
  const NodeSetList&
  getNodeSets() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }
  const NodeSetGIDsList&
  getNodeSetGIDs() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }
  const NodeSetCoordList&
  getNodeSetCoords() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get Side set lists
  const SideSetList&
  getSideSets(const int ws) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get Side set view lists
  const LocalSideSetInfoList&
  getSideSetViews(const int ws) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get local DOF views for GatherVerticallyContractedSolution
  const std::map<std::string, Kokkos::DualView<LO****, PHX::Device>>&
  getLocalDOFViews(const int workset) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get coordinates (overlap map).
  const Teuchos::ArrayRCP<double>&
  getCoordinates() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Print the coords for mesh debugging
  void
  printCoords() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get the map side_id->side_set_elem_id
  const std::map<std::string, std::map<GO, GO>>&
  getSideToSideSetCellMap() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get the map side_node_id->side_set_cell_node_id
  const std::map<std::string, std::map<GO, std::vector<int>>>&
  getSideNodeNumerationMap() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get MeshStruct
  Teuchos::RCP<AbstractMeshStruct>
  getMeshStruct() const {
    return m_mesh_struct;
  }

  //! Get nodal parameters state info struct
  const StateInfoStruct&
  getNodalParameterSIS() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Retrieve connectivity map from elementGID to workset
  WsLIDList&
  getElemGIDws() {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }
  const WsLIDList&
  getElemGIDws() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Flag if solution has a restart values -- used in Init Cond
  bool
  hasRestartSolution() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! File time of restart solution
  double
  restartDataTime() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get number of spatial dimensions
  int
  getNumDim() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Get number of total DOFs per node
  int
  getNumEq() const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  // --- Get/set solution/residual/field vectors to/from mesh --- //
  Teuchos::RCP<Thyra_Vector>
  getSolutionField(bool overlapped = false) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  Teuchos::RCP<Thyra_MultiVector>
  getSolutionMV(bool overlapped = false) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  void
  getField(Thyra_Vector& field_vector, const std::string& field_name) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }
  void
  setField(
      const Thyra_Vector& field_vector,
      const std::string&  field_name,
      bool                overlapped) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  void setFieldData(const Teuchos::RCP<StateInfoStruct>& sis) override;

  // --- Methods to write solution in the output file --- //

  //! Write the solution to the output file. Calls next two together.
  void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  } 
  void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  } 
  void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  } 
  void
  writeSolutionMV(
      const Thyra_MultiVector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double             time,
      const bool               overlapped = false,
      const bool               force_write_solution = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  } 
  //! Write the solution to the mesh database.
  void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double        time,
      const bool          overlapped = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }
  void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const double        time,
      const bool          overlapped = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }
  void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const double        time,
      const bool          overlapped = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }
  void
  writeSolutionMVToMeshDatabase(
      const Thyra_MultiVector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double             time,
      const bool               overlapped = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  }

  //! Write the solution to file. Must call writeSolution first.
  void
  writeSolutionToFile(
      const Thyra_Vector& solution,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  } 
  void
  writeSolutionMVToFile(
      const Thyra_MultiVector& solution,
      const double             time,
      const bool               overlapped = false,
      const bool               force_write_solution = false) {
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"NOT IMPLEMENTED!");
  } 

protected:

  Teuchos::RCP<OmegahAbstractMesh> m_mesh_struct;
};

}  // namespace Albany

#endif  // ALBANY_OMEGAH_DISCRETIZATION_HPP
