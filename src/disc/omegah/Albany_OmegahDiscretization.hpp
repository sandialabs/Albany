//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_OMEGAH_DISCRETIZATION_HPP
#define ALBANY_OMEGAH_DISCRETIZATION_HPP

#include "Albany_AbstractDiscretization.hpp"

#include "Albany_OmegahGenericMesh.hpp"

#include "Albany_NullSpaceUtils.hpp"

namespace Albany {

class OmegahDiscretization : public AbstractDiscretization
{
public:
  OmegahDiscretization(
      const Teuchos::RCP<Teuchos::ParameterList>& discParams,
      const int neq,
      const Teuchos::RCP<OmegahGenericMesh>&      mesh,
      const Teuchos::RCP<const Teuchos_Comm>&     comm,
      const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null,
      const std::map<int, std::vector<std::string>>& sideSetEquations =
          std::map<int, std::vector<std::string>>());

  //! Destructor
  virtual ~OmegahDiscretization() = default;

  void updateMesh () override;

  //! Get Node set lists
  const NodeSetList&
  getNodeSets() const override {
    return m_node_sets;
  }
  const NodeSetGIDsList&
  getNodeSetGIDs() const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::getNodeSetGIDs");
  }
  const NodeSetCoordList&
  getNodeSetCoords() const override {
    return m_node_set_coords;
  }

  //! Get Side set lists
  const SideSetList&
  getSideSets(const int ws) const override {
    return m_side_sets[ws];
  }

  //! Get Side set view lists
  const LocalSideSetInfoList&
  getSideSetViews(const int ws) const override {
    return m_side_set_views.at(ws);
  }

  //! Get local DOF views for GatherVerticallyContractedSolution
  const strmap_t<Kokkos::DualView<LO****, PHX::Device>>&
  getLocalDOFViews(const int workset) const override {
    return m_ws_local_dof_views.at(workset);
  }

  //! Get coordinates (overlap map).
  const Teuchos::ArrayRCP<double>& getCoordinates() const override { return m_nodes_coordinates; }

  //! Print the coords for mesh debugging
  void
  printCoords() const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::printCoords");
  }

  //! Get the map side_id->side_set_elem_id
  const std::map<std::string, std::map<GO, GO>>&
  getSideToSideSetCellMap() const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::getSideToSideSetCellMap");
  }

  //! Get the map side_node_id->side_set_cell_node_id
  const std::map<std::string, std::map<GO, std::vector<int>>>&
  getSideNodeNumerationMap() const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::getSideNodeNumerationMap");
  }

  //! Get MeshStruct
  Teuchos::RCP<AbstractMeshStruct>
  getMeshStruct() const override {
    return m_mesh_struct;
  }

  //! Retrieve connectivity map from elementGID to workset
  WsLIDList&
  getElemGIDws() override {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::getElemGIDws");
  }
  const WsLIDList&
  getElemGIDws() const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::getElemGIDws");
  }

  //! Flag if solution has a restart values -- used in Init Cond
  bool
  hasRestartSolution() const override {
    return m_mesh_struct->hasRestartSolution();
  }

  //! File time of restart solution
  double
  restartDataTime() const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::restartDataTime");
  }

  //! Get number of spatial dimensions
  int
  getNumDim() const override {
    return m_mesh_struct->meshSpecs[0]->numDim;
  }

  //! Get number of total DOFs per node
  int
  getNumEq() const override {
    return m_neq;
  }

  // --- Get/set solution/residual/field vectors to/from mesh --- //
  Teuchos::RCP<Thyra_Vector>
  getSolutionField(bool /* overlapped */ = false) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::getSolutionField");
  }

  void getSolutionMV (Thyra_MultiVector& solution, bool overlapped) const override;
  void getSolutionDxDp (Thyra_MultiVector& /* result */, bool /* overlapped */) const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahDiscretization::getSolutionDxDp");
  }

  void getField(      Thyra_Vector& field_vector,
                const std::string&  field_name) const override;

  void setField(const Thyra_Vector& field_vector,
                const std::string&  field_name,
                bool                overlapped) override;

  void setFieldData(const Teuchos::RCP<StateInfoStruct>& sis) override;

  //! Write the solution to the mesh database.
  void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                    const bool          overlapped) override;

  void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                    const Thyra_Vector& solution_dot,
                                    const bool          overlapped) override;

  void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                    const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                    const Thyra_Vector& solution_dot,
                                    const Thyra_Vector& solution_dotdot,
                                    const bool          overlapped) override;

  void writeSolutionMVToMeshDatabase (const Thyra_MultiVector& solution,
                                      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                      const bool               overlapped) override;

  //! Write the current mesh database to file
  void writeMeshDatabaseToFile (const double time,
                                const bool   force_write_solution) override;

  Teuchos::RCP<AdaptationData>
  checkForAdaptation (const Teuchos::RCP<const Thyra_Vector>& /* solution */,
                      const Teuchos::RCP<const Thyra_Vector>& /* solution_dot */,
                      const Teuchos::RCP<const Thyra_Vector>& /* solution_dotdot */,
                      const Teuchos::RCP<const Thyra_MultiVector>& /* dxdp */) const override;

  void adapt (const Teuchos::RCP<AdaptationData>& /* adaptData */) override;

protected:

  Teuchos::RCP<DOFManager>
  create_dof_mgr (const std::string& part_name,
                  const std::string& field_name,
                  const FE_Type fe_type,
                  const int order,
                  const int dof_dim) const;

  void computeNodeSets ();
  void computeGraphs ();

  // ======================= Members ======================= //

  Teuchos::RCP<Teuchos::ParameterList> m_disc_params;

  Teuchos::RCP<OmegahGenericMesh> m_mesh_struct;

  std::vector<std::string> m_sol_names;

  // Maps a Tpetra LID to the pos of a node in the omegah arrays
  std::vector<int>  m_node_lid_to_omegah_pos;

  // TODO: move stuff below in base class?
  Teuchos::RCP<const Teuchos_Comm> m_comm;

  std::map<int, LocalSideSetInfoList> m_side_set_views;
  std::vector<SideSetList> m_side_sets;
  std::map<int, strmap_t<Kokkos::DualView<LO****, PHX::Device>>> m_ws_local_dof_views;

  NodeSetList       m_node_sets;
  NodeSetCoordList  m_node_set_coords;

  // Coordinates spliced together, as [x0,y0,z0,x1,y1,z1,...]
  Teuchos::ArrayRCP<double>   m_nodes_coordinates;

  //! Equations that are defined only on some side sets of the mesh
  std::map<int, std::vector<std::string>> m_side_set_equations;

  // Number of equations (and unknowns) per node
  // TODO: this should soon be removed, in favor of more granular description of each dof/unknown
  const int m_neq;

  // TODO: I don't think this belongs in the disc class. We do store it for STK meshes,
  //       mainly b/c we need to know how many components the soln vector has.
  //       But I think we can get rid of it. In principle, we should handle time derivatives
  //       from the app/problem side.
  int m_num_time_deriv;
};

}  // namespace Albany

#endif  // ALBANY_OMEGAH_DISCRETIZATION_HPP
