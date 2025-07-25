//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STK_DISCRETIZATION_HPP
#define ALBANY_STK_DISCRETIZATION_HPP

#include <utility>
#include <vector>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Albany_NullSpaceUtils.hpp"

// Start of STK stuff
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#ifdef ALBANY_SEACAS
#include <stk_io/StkMeshIoBroker.hpp>
#endif

#include "Albany_AbstractSTKFieldContainer.hpp"

namespace Albany {

// ====================== STK Discretization ===================== //

class STKDiscretization : public AbstractDiscretization
{
public:
  //! Constructor
  STKDiscretization(
      const Teuchos::RCP<Teuchos::ParameterList>& discParams,
      const int neq,
      Teuchos::RCP<AbstractSTKMeshStruct>&        stkMeshStruct,
      const Teuchos::RCP<const Teuchos_Comm>&     comm,
      const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null,
      const std::map<int, std::vector<std::string>>& sideSetEquations =
          std::map<int, std::vector<std::string>>());

  //! Destructor
  virtual ~STKDiscretization();

  void
  printConnectivity() const;

  //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
  const NodeSetList&
  getNodeSets() const
  {
    return nodeSets;
  }
  const NodeSetGIDsList&
  getNodeSetGIDs() const
  {
    return nodeSetGIDs;
  }
  const NodeSetCoordList&
  getNodeSetCoords() const
  {
    return nodeSetCoords;
  }

  //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
  const SideSetList&
  getSideSets(const int workset) const
  {
    return sideSets[workset];
  }

  //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
  const LocalSideSetInfoList&
  getSideSetViews(const int workset) const
  {
    return sideSetViews.at(workset);
  }

  //! Get local DOF views for GatherVerticallyContractedSolution
  const std::map<std::string, Kokkos::DualView<LO****, PHX::Device>>&
  getLocalDOFViews(const int workset) const
  {
    return wsLocalDOFViews.at(workset);
  }

  //! Get connectivity map from elementGID to workset
  WsLIDList&
  getElemGIDws()
  {
    return elemGIDws;
  }
  WsLIDList const&
  getElemGIDws() const
  {
    return elemGIDws;
  }

  //! Retrieve coordinate vector (num_used_nodes * 3)
  const Teuchos::ArrayRCP<double>&
  getCoordinates() const;

  //! Print the coordinates for debugging
  void
  printCoords() const;

  // Retrieve mesh struct
  Teuchos::RCP<AbstractSTKMeshStruct>
  getSTKMeshStruct() const
  {
    return stkMeshStruct;
  }
  Teuchos::RCP<AbstractMeshStruct>
  getMeshStruct() const
  {
    return stkMeshStruct;
  }

  const std::map<std::string, std::map<GO, GO>>&
  getSideToSideSetCellMap() const
  {
    return sideToSideSetCellMap;
  }

  const std::map<std::string, std::map<GO, std::vector<int>>>&
  getSideNodeNumerationMap() const
  {
    return sideNodeNumerationMap;
  }

  //! Flag if solution has a restart values -- used in Init Cond
  bool
  hasRestartSolution() const
  {
    return stkMeshStruct->hasRestartSolution();
  }

  //! If restarting, convenience function to return restart data time
  double
  restartDataTime() const
  {
    return stkMeshStruct->restartDataTime();
  }

  //! After mesh modification, need to update the element connectivity and nodal
  //! coordinates
  void
  updateMesh();

  //! Function that transforms an STK mesh of a unit cube (for LandIce problems)
  void
  transformMesh();

  //! Get number of spatial dimensions
  int
  getNumDim() const
  {
    return stkMeshStruct->numDim;
  }

  //! Get number of total DOFs per node
  int
  getNumEq() const
  {
    return neq;
  }

  const stk::mesh::MetaData&
  getSTKMetaData() const
  {
    return *metaData;
  }
  const stk::mesh::BulkData&
  getSTKBulkData() const
  {
    return *bulkData;
  }

  // Used very often, so make it a function
  GO stk_gid (const stk::mesh::Entity e) const {
    // STK numbering is 1-based, while we want 0-based.
    return getSTKBulkData().identifier(e) - 1;
  }

  // --- Get/set solution/residual/field vectors to/from mesh --- //

  Teuchos::RCP<Thyra_Vector>
  getSolutionField(const bool overlapped = false) const;

  void
  getSolutionMV(Thyra_MultiVector& result, bool overlapped) const;

  void
  getSolutionDxDp (Thyra_MultiVector& result, bool overlapped) const;

  void
  getField(Thyra_Vector& field_vector, const std::string& field_name) const;
  void
  setField(
      const Thyra_Vector& field_vector,
      const std::string&  field_name,
      const bool          overlapped = false);

  //! Write the solution to the mesh database.
  void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const bool overlapped);
  void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const bool overlapped);
  void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const bool overlapped);
  void
  writeSolutionMVToMeshDatabase(
      const Thyra_MultiVector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const bool overlapped);

  //! Write the solution to file. Must call writeSolution first.
  void
  writeMeshDatabaseToFile(
      const double        time,
      const bool          force_write_solution) override;

  Teuchos::RCP<AdaptationData>
  checkForAdaptation (const Teuchos::RCP<const Thyra_Vector>& solution,
                      const Teuchos::RCP<const Thyra_Vector>& solution_dot,
                      const Teuchos::RCP<const Thyra_Vector>& solution_dotdot,
                      const Teuchos::RCP<const Thyra_MultiVector>& dxdp) override;

  void adapt (const Teuchos::RCP<AdaptationData>& adaptData) override;

  /** Add a solution field
    */
  void addSolutionField(const std::string & fieldName,const std::string & blockId);

  /** Add a solution field
    */
  void addCellField(const std::string & fieldName,const std::string & blockId);

  void setFieldData();

  Teuchos::RCP<AbstractSTKFieldContainer> getSolutionFieldContainer() {
    return solutionFieldContainer;
  }

  //! Find the local position of child entity within parent entity
  int determine_entity_pos (const stk::mesh::Entity parent,
                            const stk::mesh::Entity child) const;

 protected:

  friend class BlockedSTKDiscretization;
  friend class STKConnManager;

  void
  getSolutionField(Thyra_Vector& result, bool overlapped) const;

  double
  monotonicTimeLabel(const double time);

  void computeVectorSpaces();

  //! Process STK mesh for CRS Graphs
  virtual void
  computeGraphs();
  //! Process coords for ML
  void
  setupMLCoords();
  //! Process STK mesh for Workset/Bucket Info
  void
  computeWorksetInfo();
  //! Process STK mesh for NodeSets
  void
  computeNodeSets();
  //! Process STK mesh for SideSets
  void
  computeSideSets();
  //! Call stk_io for creating exodus output file
  void
  setupExodusOutput();

  void
  writeCoordsToMatrixMarket() const;

  void
  buildSideSetProjectors();

  double previous_time_label;

  // If node_as_elements=true, build the ConnMgr as if nodes are the "cells".
  Teuchos::RCP<DOFManager>
  create_dof_mgr (const std::string& part_name,
                  const std::string& field_name,
                  const FE_Type fe_type,
                  const int order,
                  const int dof_dim) const;

  // ==================== Members =================== //

  Teuchos::RCP<Teuchos::FancyOStream> out;

  //! Stk Mesh Objects
  Teuchos::RCP<stk::mesh::MetaData> metaData;
  Teuchos::RCP<stk::mesh::BulkData> bulkData;

  //! Teuchos communicator
  Teuchos::RCP<const Teuchos_Comm> comm;

  //! Number of equations (and unknowns) per node
  const int neq;

  //! Equations that are defined only on some side sets of the mesh
  std::map<int, std::vector<std::string>> sideSetEquations;

  //! Number of elements on this processor
  unsigned int numMyElements;

  //! node sets stored as std::map(string ID, int vector of GIDs)
  NodeSetList      nodeSets;
  NodeSetGIDsList  nodeSetGIDs;
  NodeSetCoordList nodeSetCoords;

  //! side sets stored as std::map(string ID, SideArray classes) per workset
  //! (std::vector across worksets)
  std::vector<SideSetList> sideSets;
  GlobalSideSetList globalSideSetViews;
  std::map<int, LocalSideSetInfoList> sideSetViews;

  //! GatherVerticallyContractedSolution connectivity
  std::map<std::string, Kokkos::DualView<LO****, PHX::Device>> allLocalDOFViews;
  std::map<int, std::map<std::string, Kokkos::DualView<LO****, PHX::Device>>> wsLocalDOFViews;

  mutable Teuchos::ArrayRCP<double>                                 coordinates;
  Teuchos::RCP<Thyra_MultiVector>                                   coordMV;

  //! Connectivity map from elementGID to workset and LID in workset
  WsLIDList elemGIDws;

  // States: vector of length worksets of a map from field name to shards array
  std::vector<std::vector<std::vector<double>>> nodesOnElemStateVec;

  //! Number of elements on this processor
  GO  maxGlobalNodeGID;

  // Needed to pass coordinates to ML.
  Teuchos::RCP<RigidBodyModes> rigidBodyModes;

  // Storage used in periodic BCs to un-roll coordinates. Pointers saved for
  // destructor.
  std::vector<double*> toDelete;

  Teuchos::RCP<AbstractSTKMeshStruct> stkMeshStruct;

  Teuchos::RCP<Teuchos::ParameterList> discParams;

  // Sideset discretizations
  std::map<std::string, Teuchos::RCP<STKDiscretization>> sideSetDiscretizationsSTK;
  std::map<std::string, std::map<GO, GO>>                sideToSideSetCellMap;
  std::map<std::string, std::map<GO, std::vector<int>>>  sideNodeNumerationMap;
  std::map<std::string, Teuchos::RCP<Thyra_LinearOp>>    projectors;
  std::map<std::string, Teuchos::RCP<Thyra_LinearOp>>    ov_projectors;

// Used in Exodus writing capability
#ifdef ALBANY_SEACAS
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data;

  int outputInterval;

  size_t outputFileIdx;
#endif

  Teuchos::RCP<AbstractSTKFieldContainer> solutionFieldContainer;
};

}  // namespace Albany

#endif  // ALBANY_STK_DISCRETIZATION_HPP
