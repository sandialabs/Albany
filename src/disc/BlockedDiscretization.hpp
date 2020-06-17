//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef BLOCKED_DISCRETIZATION_HPP
#define BLOCKED_DISCRETIZATION_HPP

#include <utility>
#include <vector>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_DataTypes.hpp"
#include "utility/Albany_ThyraBlockedCrsMatrixFactory.hpp"
#include "utility/Albany_ThyraUtils.hpp"

#include "Albany_NullSpaceUtils.hpp"

namespace Albany {

// This BlockedDiscretization class implements multiple discretization blocks and serves them up as Thyra_ProductVectors.
// In this implementation, each block has to be the same, and the discretization for each block is defined here

using BlckDisc = STKDiscretization;

class BlockedDiscretization : public AbstractDiscretization
{

 public:
  //! Constructor
  BlockedDiscretization(
      const Teuchos::RCP<Teuchos::ParameterList>& discParams,
      Teuchos::RCP<AbstractSTKMeshStruct>&        stkMeshStruct,
      const Teuchos::RCP<const Teuchos_Comm>&     comm,
      const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null,
      const std::map<int, std::vector<std::string>>& sideSetEquations =
          std::map<int, std::vector<std::string>>());

  //! Destructor
  virtual ~BlockedDiscretization();

  void
  printConnectivity() const;

  //! Get node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace>
  getNodeVectorSpace() const
  {
    return BlockDiscretization[0]->getNodeVectorSpace();
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapNodeVectorSpace() const
  {
    return BlockDiscretization[0]->getOverlapNodeVectorSpace();
  }

  //! Get solution DOF vector space (owned and overlapped).
  Teuchos::RCP<const Thyra_VectorSpace>
  getVectorSpace() const
  {
    return BlockDiscretization[0]->getVectorSpace();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapVectorSpace() const
  {
    return BlockDiscretization[0]->getOverlapVectorSpace();
  }

  //! Get Field node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace>
  getNodeVectorSpace(const std::string& field_name) const;
  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapNodeVectorSpace(const std::string& field_name) const;

  //! Get Field vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace>
  getVectorSpace(const std::string& field_name) const;
  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapVectorSpace(const std::string& field_name) const;

  //! Create a Jacobian operator (owned and overlapped)
  Teuchos::RCP<Thyra_LinearOp>
  createJacobianOp() const
  {
    return m_jac_factory->createOp();
  }
  Teuchos::RCP<Thyra_LinearOp>
  createOverlapJacobianOp() const
  {
    return m_overlap_jac_factory->createOp();
  }

  //! Get node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_ProductVectorSpace>
  getNodeProductVectorSpace() const
  {
    return m_node_pvs;
  }
  Teuchos::RCP<const Thyra_ProductVectorSpace>
  getOverlapNodeProductVectorSpace() const
  {
    return m_overlap_node_pvs;
  }

  //! Get solution DOF vector space (owned and overlapped).
  Teuchos::RCP<const Thyra_ProductVectorSpace>
  getProductVectorSpace() const
  {
    return m_pvs;
  }

  Teuchos::RCP<const Thyra_ProductVectorSpace>
  getOverlapProductVectorSpace() const
  {
    return m_overlap_pvs;
  }

  //! Get Field node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_ProductVectorSpace>
  getNodeProductVectorSpace(const std::string& field_name) const;
  Teuchos::RCP<const Thyra_ProductVectorSpace>
  getOverlapNodeProductVectorSpace(const std::string& field_name) const;

  //! Get Field vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_ProductVectorSpace>
  getProductVectorSpace(const std::string& field_name) const;
  Teuchos::RCP<const Thyra_ProductVectorSpace>
  getOverlapProductVectorSpace(const std::string& field_name) const;

  //! Create a Jacobian operator (owned and overlapped)
  Teuchos::RCP<Thyra_BlockedLinearOp>
  createBlockedJacobianOp() const
  {
    return m_jac_factory->createOp();
  }

  Teuchos::RCP<Thyra_BlockedLinearOp>
  createOverlapBlockedJacobianOp() const
  {
    return m_overlap_jac_factory->createOp();
  }

  bool
  isExplicitScheme() const
  {
    return false;
  }

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

  //! Get map from ws, elem, node [, eq] -> [Node|DOF] GID
  const Conn&
  getWsElNodeEqID() const
  {
    return wsElNodeEqID;
  }

  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>::type&
  getWsElNodeID() const
  {
    return wsElNodeID;
  }

  //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for
  //! both scalar and vector fields
  const std::vector<IDArray>&
  getElNodeEqID(const std::string& field_name) const
  {
    return BlockDiscretization[0]->getElNodeEqID(field_name);
  }

  const NodalDOFManager&
  getDOFManager(const std::string& field_name) const
  {
    return BlockDiscretization[0]->getDOFManager(field_name);
  }

  const NodalDOFManager&
  getOverlapDOFManager(const std::string& field_name) const
  {
    return BlockDiscretization[0]->getOverlapDOFManager(field_name);
  }

  //! Retrieve coodinate vector (num_used_nodes * 3)
  const Teuchos::ArrayRCP<double>&
  getCoordinates() const;
  void
  setCoordinates(const Teuchos::ArrayRCP<const double>& c);

  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>::type&
  getCoords() const
  {
    return coords;
  }

  //! Print the coordinates for debugging
  void
  printCoords() const;

  //! Set stateArrays
  void
  setStateArrays(StateArrays& sa)
  {
    stateArrays = sa;
  }

  //! Get stateArrays
  StateArrays&
  getStateArrays()
  {
    return stateArrays;
  }

  //! Get nodal parameters state info struct
  const StateInfoStruct&
  getNodalParameterSIS() const
  {
    return stkMeshStruct->getFieldContainer()->getNodalParameterSIS();
  }

  //! Retrieve Vector (length num worksets) of element block names
  const WorksetArray<std::string>::type&
  getWsEBNames() const
  {
    return wsEBNames;
  }
  //! Retrieve Vector (length num worksets) of physics set index
  const WorksetArray<int>::type&
  getWsPhysIndex() const
  {
    return wsPhysIndex;
  }

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

  const SideSetDiscretizationsType&
  getSideSetDiscretizations() const
  {
    return sideSetDiscretizations;
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

  //! Close current exodus file in stk_io and create a new one for an adapted
  //! mesh and new results
  void
  reNameExodusOutput(std::string& filename);

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
    return BlockDiscretization[0]->getNumEq();
  }

  //! Locate nodal dofs in non-overlapping vectors using local indexing
  int
  getOwnedDOF(const int inode, const int eq) const;

  //! Locate nodal dofs in overlapping vectors using local indexing
  int
  getOverlapDOF(const int inode, const int eq) const;

  //! Get global id of the stk entity
  GO
  gid(const stk::mesh::Entity entity) const;

  //! Locate nodal dofs using global indexing
  GO
  getGlobalDOF(const GO inode, const int eq) const;

  Teuchos::RCP<LayeredMeshNumbering<LO>>
  getLayeredMeshNumbering() const
  {
    return stkMeshStruct->layered_mesh_numbering;
  }

  const stk::mesh::MetaData&
  getSTKMetaData() const
  {
    return BlockDiscretization[0]->getSTKMetaData();
  }
  const stk::mesh::BulkData&
  getSTKBulkData() const
  {
    return BlockDiscretization[0]->getSTKBulkData();
  }

  // --- Get/set solution/residual/field vectors to/from mesh --- //

  Teuchos::RCP<Thyra_MultiVector>
  getBlockedSolutionField(const bool overlapped = false) const;
  // These are identical for a blocked discretization, keep the below to keep the interface the same
  Teuchos::RCP<Thyra_MultiVector>
  getBlockedSolutionMV(const bool overlapped = false) const;

  void
  getField(Thyra_MultiVector& field_vector, const std::string& field_name) const;
  void
  setField(
      const Thyra_MultiVector& field_vector,
      const std::string&  field_name,
      const bool          overlapped = false);

  // --- Methods to write solution in the output file --- //

  void
  writeSolution(
      const Thyra_MultiVector& solution,
      const double        time,
      const bool          overlapped = false);
  void
  writeSolution(
      const Thyra_MultiVector& solution,
      const Thyra_MultiVector& solution_dot,
      const double        time,
      const bool          overlapped = false);
  void
  writeSolution(
      const Thyra_MultiVector& solution,
      const Thyra_MultiVector& solution_dot,
      const Thyra_MultiVector& solution_dotdot,
      const double        time,
      const bool          overlapped = false);
  void
  writeSolutionMV(
      const Thyra_MultiVector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double             time,
      const bool               overlapped = false);

  //! Write the solution to the mesh database.
  void
  writeSolutionToMeshDatabase(
      const Thyra_MultiVector& solution,
      const double /* time */,
      const bool overlapped = false);
  void
  writeSolutionToMeshDatabase(
      const Thyra_MultiVector& solution,
      const Thyra_MultiVector& solution_dot,
      const double /* time */,
      const bool overlapped = false);
  void
  writeSolutionToMeshDatabase(
      const Thyra_MultiVector& solution,
      const Thyra_MultiVector& solution_dot,
      const Thyra_MultiVector& solution_dotdot,
      const double /* time */,
      const bool overlapped = false);
  void
  writeSolutionMVToMeshDatabase(
      const Thyra_MultiVector& solution,
      const double /* time */,
      const bool overlapped = false);

  //! Write the solution to file. Must call writeSolution first.
  void
  writeSolutionToFile(
      const Thyra_MultiVector& solution,
      const double        time,
      const bool          overlapped = false);
  void
  writeSolutionMVToFile(
      const Thyra_MultiVector& solution,
      const double             time,
      const bool               overlapped = false);

  //! used when NetCDF output on a latitude-longitude grid is requested.
  // Each struct contains a latitude/longitude index and it's parametric
  // coordinates in an element.
  struct interp
  {
    std::pair<double, double>     parametric_coords;
    std::pair<unsigned, unsigned> latitude_longitude;
  };

  // ==================== Members =================== //

  Teuchos::RCP<Teuchos::FancyOStream> out;

  //! Teuchos communicator
  Teuchos::RCP<const Teuchos_Comm> comm;

  //! Unknown map and node map
  Teuchos::RCP<const Thyra_ProductVectorSpace> m_pvs;
  Teuchos::RCP<const Thyra_ProductVectorSpace> m_node_pvs;

  //! Overlapped unknown map and node map
  Teuchos::RCP<const Thyra_ProductVectorSpace> m_overlap_pvs;
  Teuchos::RCP<const Thyra_ProductVectorSpace> m_overlap_node_pvs;

  //! Jacobian matrix graph proxy (owned and overlap)
  Teuchos::RCP<ThyraBlockedCrsMatrixFactory> m_jac_factory;
  Teuchos::RCP<ThyraBlockedCrsMatrixFactory> m_overlap_jac_factory;

  //! Processor ID
  unsigned int myPID;

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

  //! Connectivity array [workset, element, local-node, Eq] => LID
  Conn wsElNodeEqID;

  //! Connectivity array [workset, element, local-node] => GID
  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>::type wsElNodeID;

  mutable Teuchos::ArrayRCP<double>                                 coordinates;
  Teuchos::RCP<Thyra_MultiVector>                                   coordMV;
  WorksetArray<std::string>::type                                   wsEBNames;
  WorksetArray<int>::type                                           wsPhysIndex;
  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>::type coords;
  WorksetArray<Teuchos::ArrayRCP<double>>::type  sphereVolume;
  WorksetArray<Teuchos::ArrayRCP<double*>>::type latticeOrientation;

  //! Connectivity map from elementGID to workset and LID in workset
  WsLIDList elemGIDws;

  // States: vector of length worksets of a map from field name to shards array
  StateArrays                                   stateArrays;
  std::vector<std::vector<std::vector<double>>> nodesOnElemStateVec;

  //! list of all owned nodes, saved for setting solution
  std::vector<stk::mesh::Entity> ownednodes;
  std::vector<stk::mesh::Entity> cells;

  //! list of all overlap nodes, saved for getting coordinates for mesh motion
  std::vector<stk::mesh::Entity> overlapnodes;

  //! Number of elements on this processor
  int numOwnedNodes;
  int numOverlapNodes;
  GO  maxGlobalNodeGID;

  // Needed to pass coordinates to ML.
  Teuchos::RCP<RigidBodyModes> rigidBodyModes;

  int              netCDFp;
  size_t           netCDFOutputRequest;
  std::vector<int> varSolns;

  // Storage used in periodic BCs to un-roll coordinates. Pointers saved for
  // destructor.
  std::vector<double*> toDelete;

  Teuchos::RCP<AbstractSTKMeshStruct> stkMeshStruct;

  Teuchos::RCP<Teuchos::ParameterList> discParams;

  // Sideset discretizations
  std::map<std::string, Teuchos::RCP<AbstractDiscretization>>
      sideSetDiscretizations;
  std::map<std::string, std::map<GO, GO>>               sideToSideSetCellMap;
  std::map<std::string, std::map<GO, std::vector<int>>> sideNodeNumerationMap;
  std::map<std::string, Teuchos::RCP<Thyra_BlockedLinearOp>>   projectors;
  std::map<std::string, Teuchos::RCP<Thyra_BlockedLinearOp>>   ov_projectors;

// Used in Exodus writing capability
#ifdef ALBANY_SEACAS
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data;

  int outputInterval;

  size_t outputFileIdx;
#endif
  DiscType interleavedOrdering;

 private:

  Teuchos::Array<Teuchos::RCP<BlckDisc> > BlockDiscretization;

  Teuchos::RCP<ThyraBlockedCrsMatrixFactory> nodalMatrixFactory;

};

}  // namespace Albany

#endif  // BLOCKED_DISCRETIZATION_HPP
