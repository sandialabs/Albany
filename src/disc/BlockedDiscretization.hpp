//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_BLOCKED_DISCRETIZATION_HPP
#define ALBANY_BLOCKED_DISCRETIZATION_HPP

#include <utility>
#include <vector>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_DataTypes.hpp"
#include "utility/Albany_ThyraBlockedCrsMatrixFactory.hpp"
#include "utility/Albany_ThyraUtils.hpp"

//#include "Albany_NullSpaceUtils.hpp"

namespace Albany {

// This BlockedDiscretization class implements multiple discretization blocks and serves them up as Thyra_ProductVectors.
// In this implementation, each block has to be the same, and the discretization for each block is defined here

class BlockedDiscretization : public AbstractDiscretization
{
public:
  // The type of discretization stored internally. For now, only STK.
  // TODO: generalize
  using disc_type = STKDiscretization;

  //! Constructor
  BlockedDiscretization(
      const Teuchos::RCP<Teuchos::ParameterList>& discParams,
      Teuchos::RCP<AbstractSTKMeshStruct>&        stkMeshStruct,
      const Teuchos::RCP<const Teuchos_Comm>&     comm,
      const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null,
      const std::map<int, std::vector<std::string>>& sideSetEquations =
          std::map<int, std::vector<std::string>>());

  //! Destructor
  virtual ~BlockedDiscretization() = default;

  /** Does a fieldOrder string require blocking? 
    * A field order is basically stetup like this
    *    blocked: <field 0> <field 1> 
    * where two blocks will be created. To merge fields
    * between blocks use a hyphen, i.e.
    *    blocked: <field 0> <field 1> - <field 2> - <field 3>
    * This will create 2 blocks, the first contains only <field 0>
    * and the second combines <field 1>, <field 2> and <field 3>. Note
    * the spaces before and after the hyphen, these are important!
    */

  static bool requiresBlocking(const std::string & fieldorder);

  static void buildBlocking(const std::string & fieldorder,std::vector<std::vector<std::string> > & blocks);

  void
  printConnectivity() const;

  //! Get node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace>
  getNodeVectorSpace() const
  {
    return m_blocks[0]->getNodeVectorSpace();
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapNodeVectorSpace() const
  {
    return m_blocks[0]->getOverlapNodeVectorSpace();
  }

  //! Get solution DOF vector space (owned and overlapped).
  Teuchos::RCP<const Thyra_VectorSpace>
  getVectorSpace() const
  {
    return m_blocks[0]->getVectorSpace();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapVectorSpace() const
  {
    return m_blocks[0]->getOverlapVectorSpace();
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
    return m_blocks[0]->getSideSets(workset);
  }

  //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
  const LocalSideSetInfoList&
  getSideSetViews(const int workset) const
  {
    return m_blocks[0]->getSideSetViews(workset);
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
    return m_blocks[0]->getElNodeEqID(field_name);
  }

  const NodalDOFManager&
  getDOFManager(const std::string& field_name) const
  {
    return m_blocks[0]->getDOFManager(field_name);
  }

  const NodalDOFManager&
  getOverlapDOFManager(const std::string& field_name) const
  {
    return m_blocks[0]->getOverlapDOFManager(field_name);
  }

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
    return m_blocks[0]->getNumEq();
  }

  Teuchos::RCP<LayeredMeshNumbering<GO>>
  getLayeredMeshNumbering() const
  {
    return stkMeshStruct->layered_mesh_numbering;
  }

  const stk::mesh::MetaData&
  getSTKMetaData() const
  {
    return m_blocks[0]->getSTKMetaData();
  }
  const stk::mesh::BulkData&
  getSTKBulkData() const
  {
    return m_blocks[0]->getSTKBulkData();
  }

  // --- Get/set solution/residual/field vectors to/from mesh --- //

  Teuchos::RCP<Thyra_MultiVector>
  getBlockedSolutionField(const bool overlapped = false) const;
  // These are identical for a blocked discretization, keep the below to keep the interface the same
  Teuchos::RCP<Thyra_MultiVector>
  getBlockedSolutionMV(const bool overlapped = false) const;

// GAH - These all need a serious look, they are just stubbed in to get things compiling

  //! Retrieve coordinate vector (num_used_nodes * 3)
  const Teuchos::ArrayRCP<double>&
  getCoordinates() const {
     return m_blocks[0]->getCoordinates();
  }

  Teuchos::RCP<Thyra_Vector>
  getSolutionField(bool overlapped) const
  {
    return m_blocks[0]->getSolutionField(overlapped);
  }

  Teuchos::RCP<Thyra_MultiVector>
  getSolutionMV(bool overlapped) const
  {
    return m_blocks[0]->getSolutionMV(overlapped);
  }

  void
  getField(Thyra_Vector& result, const std::string& name) const
  {
    m_blocks[0]->getField(result, name);
  }

  void
  getSolutionField(Thyra_Vector& result, const bool overlapped) const
  {
    m_blocks[0]->getSolutionField(result, overlapped);
  }

  void
  setField(
    const Thyra_Vector& result,
    const std::string&  name,
    bool                overlapped)
  {
    m_blocks[0]->setField(result, name, overlapped);
  }

  void
  writeSolution(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double        time,
    const bool          overlapped)
  {
    m_blocks[0]->writeSolution(soln, soln_dxdp, time, overlapped);
  }

  void
  writeSolution(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const double        time,
    const bool          overlapped)
  {
    m_blocks[0]->writeSolution(soln, soln_dxdp, soln_dot, time, overlapped);
  }

  void
  writeSolution(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const double        time,
    const bool          overlapped)
  {
    m_blocks[0]->writeSolution(soln, soln_dxdp, soln_dot, soln_dotdot, time, overlapped);
  }

  void
  writeSolutionMV(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double             time,
    const bool               overlapped)
  {
    m_blocks[0]->writeSolutionMV(soln, soln_dxdp, time, overlapped);
  }

  void
  writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double /* time */,
    const bool overlapped)
  {
    m_blocks[0]->writeSolutionToMeshDatabase(soln, soln_dxdp, 0.0, overlapped);
  }

  void
  writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const double /* time */,
    const bool overlapped)
  { 
    m_blocks[0]->writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, 0.0, overlapped);
  }

  void
  writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const double /* time */,
    const bool overlapped)
  {
    m_blocks[0]->writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, soln_dotdot, 0.0, overlapped);
  }

  void
  writeSolutionMVToMeshDatabase(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double /* time */,
    const bool overlapped)
  { 
    m_blocks[0]->writeSolutionMVToMeshDatabase(soln, soln_dxdp, 0.0, overlapped);
  }

  void
  writeSolutionToFile(
    const Thyra_Vector& soln,
    const double        time,
    const bool          overlapped)
  {
    m_blocks[0]->writeSolutionToFile(soln, time, overlapped);
  }

  void
  writeSolutionMVToFile(
    const Thyra_MultiVector& soln,
    const double             time,
    const bool               overlapped)
  {
    m_blocks[0]->writeSolutionMVToFile(soln, time, overlapped);
  }

  Teuchos::RCP<const GlobalLocalIndexer>
  getGlobalLocalIndexer(const std::string& field_name) const
  {
    return m_blocks[0]->getGlobalLocalIndexer(field_name);
  }

  Teuchos::RCP<const GlobalLocalIndexer>
  getOverlapGlobalLocalIndexer(const std::string& field_name) const
  {
    return m_blocks[0]->getOverlapGlobalLocalIndexer(field_name);
  }


// GAH - End serious look

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

  //! Equations that are defined only on some side sets of the mesh
  std::map<int, std::vector<std::string>> sideSetEquations;

  //! Number of elements on this processor
  unsigned int numMyElements;

  //! node sets stored as std::map(string ID, int vector of GIDs)
  NodeSetList      nodeSets;
  NodeSetGIDsList  nodeSetGIDs;
  NodeSetCoordList nodeSetCoords;

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

  Teuchos::RCP<AbstractSTKMeshStruct> stkMeshStruct;

  Teuchos::RCP<Teuchos::ParameterList> discParams;

  // Sideset discretizations
  std::map<std::string, Teuchos::RCP<AbstractDiscretization>>
      sideSetDiscretizations;
  std::map<std::string, std::map<GO, GO>>               sideToSideSetCellMap;
  std::map<std::string, std::map<GO, std::vector<int>>> sideNodeNumerationMap;
  std::map<std::string, Teuchos::RCP<Thyra_BlockedLinearOp>>   projectors;
  std::map<std::string, Teuchos::RCP<Thyra_BlockedLinearOp>>   ov_projectors;

 private:

  Teuchos::Array<Teuchos::RCP<disc_type> > m_blocks;

  Teuchos::RCP<ThyraBlockedCrsMatrixFactory> nodalMatrixFactory;

};

}  // namespace Albany

#endif  // ALBANY_BLOCKED_DISCRETIZATION_HPP
