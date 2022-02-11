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
#include "Albany_ThyraCrsMatrixFactory.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "Albany_NullSpaceUtils.hpp"

// Start of STK stuff
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/parallel/Parallel.hpp>
#ifdef ALBANY_SEACAS
#include <stk_io/StkMeshIoBroker.hpp>
#endif

#include "Albany_AbstractSTKFieldContainer.hpp"

namespace Albany {

typedef shards::Array<GO, shards::NaturalOrder> GIDArray;

struct DOFsStruct
{
  Teuchos::RCP<const Thyra_VectorSpace> node_vs;
  Teuchos::RCP<const Thyra_VectorSpace> overlap_node_vs;
  Teuchos::RCP<const Thyra_VectorSpace> vs;
  Teuchos::RCP<const Thyra_VectorSpace> overlap_vs;
  NodalDOFManager                       dofManager;
  NodalDOFManager                       overlap_dofManager;
  std::vector<std::vector<LO>>          wsElNodeEqID_rawVec;
  std::vector<IDArray>                  wsElNodeEqID;
  std::vector<std::vector<GO>>          wsElNodeID_rawVec;
  std::vector<GIDArray>                 wsElNodeID;

  Teuchos::RCP<const GlobalLocalIndexer> node_vs_indexer;
  Teuchos::RCP<const GlobalLocalIndexer> overlap_node_vs_indexer;
  Teuchos::RCP<const GlobalLocalIndexer> vs_indexer;
  Teuchos::RCP<const GlobalLocalIndexer> overlap_vs_indexer;
};

struct NodalDOFsStructContainer
{
  typedef std::map<std::pair<std::string, int>, DOFsStruct> MapOfDOFsStructs;

  MapOfDOFsStructs                                        mapOfDOFsStructs;
  std::map<std::string, MapOfDOFsStructs::const_iterator> fieldToMap;

  const DOFsStruct&
  getDOFsStruct(const std::string& field_name) const
  {
    const auto iter = fieldToMap.find(field_name);
    if (iter == fieldToMap.end())
      TEUCHOS_TEST_FOR_EXCEPTION(true,
          std::logic_error, field_name + " does not exist in fieldToMap");
    return iter->second->second;
  };

  // IKT: added the following function, which may be useful for debugging.
  void
  printFieldToMap() const
  {
    typedef std::map<std::string, MapOfDOFsStructs::const_iterator>::
        const_iterator                  MapIterator;
    Teuchos::RCP<Teuchos::FancyOStream> out =
        Teuchos::VerboseObjectBase::getDefaultOStream();
    for (MapIterator iter = fieldToMap.begin(); iter != fieldToMap.end();
         iter++) {
      std::string key = iter->first;
      *out << "IKT Key: " << key << "\n";
      auto vs = getDOFsStruct(key).vs;
      *out << "IKT Vector Space \n: ";
      describe(vs, *out, Teuchos::VERB_EXTREME);
    }
  }

  void
  addEmptyDOFsStruct(
      const std::string& field_name,
      const std::string& meshPart,
      int                numComps)
  {
    if (numComps != 1)
      mapOfDOFsStructs.insert(make_pair(make_pair(meshPart, 1), DOFsStruct()));

    fieldToMap[field_name] =
        mapOfDOFsStructs
            .insert(make_pair(make_pair(meshPart, numComps), DOFsStruct()))
            .first;
  }
};

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

  //! Get node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace>
  getNodeVectorSpace() const
  {
    return m_node_vs;
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapNodeVectorSpace() const
  {
    return m_overlap_node_vs;
  }

  //! Get solution DOF vector space (owned and overlapped).
  Teuchos::RCP<const Thyra_VectorSpace>
  getVectorSpace() const
  {
    return m_vs;
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapVectorSpace() const
  {
    return m_overlap_vs;
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

  //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
  const LocalSideSetInfoList&
  getSideSetViews(const int workset) const
  {
    return sideSetViews.at(workset);
  }

  //! Get local DOF views for GatherVerticallyContractedSolution
  const std::map<std::string, Kokkos::View<LO****, PHX::Device>>&
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

  //! Get map from ws, elem, node [, eq] -> [Node|DOF] GID
  const Conn&
  getWsElNodeEqID() const
  {
    return wsElNodeEqID;
  }

  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>&
  getWsElNodeID() const
  {
    return wsElNodeID;
  }

  //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for
  //! both scalar and vector fields
  const std::vector<IDArray>&
  getElNodeEqID(const std::string& field_name) const
  {
    return nodalDOFsStructContainer.getDOFsStruct(field_name).wsElNodeEqID;
  }

  Teuchos::RCP<const GlobalLocalIndexer>
  getGlobalLocalIndexer(const std::string& field_name) const
  {
    return nodalDOFsStructContainer.getDOFsStruct(field_name).vs_indexer;
  }

  Teuchos::RCP<const GlobalLocalIndexer>
  getOverlapGlobalLocalIndexer(const std::string& field_name) const
  {
    return nodalDOFsStructContainer.getDOFsStruct(field_name)
        .overlap_vs_indexer;
  }

  const NodalDOFManager&
  getDOFManager(const std::string& field_name) const
  {
    return nodalDOFsStructContainer.getDOFsStruct(field_name).dofManager;
  }

  const NodalDOFManager&
  getOverlapDOFManager(const std::string& field_name) const
  {
    return nodalDOFsStructContainer.getDOFsStruct(field_name)
        .overlap_dofManager;
  }

  //! Retrieve coodinate vector (num_used_nodes * 3)
  const Teuchos::ArrayRCP<double>&
  getCoordinates() const;

  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>&
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
  const WorksetArray<std::string>&
  getWsEBNames() const
  {
    return wsEBNames;
  }
  //! Retrieve Vector (length num worksets) of physics set index
  const WorksetArray<int>&
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
    return neq;
  }

  int
  getFADLength() const
  {
    return neq * wsElNodeID[0][0].size();
  }

  Teuchos::RCP<LayeredMeshNumbering<GO>>
  getLayeredMeshNumbering() const
  {
    return stkMeshStruct->layered_mesh_numbering;
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
  Teuchos::RCP<Thyra_MultiVector>
  getSolutionMV(const bool overlapped = false) const;

  void
  getField(Thyra_Vector& field_vector, const std::string& field_name) const;
  void
  setField(
      const Thyra_Vector& field_vector,
      const std::string&  field_name,
      const bool          overlapped = false);

  // --- Methods to write solution in the output file --- //

  void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false); 
  void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false); 
  void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false); 
  void
  writeSolutionMV(
      const Thyra_MultiVector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double             time,
      const bool               overlapped = false,
      const bool               force_write_solution = false); 

  //! Write the solution to the mesh database.
  void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double /* time */,
      const bool overlapped = false);
  void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const double /* time */,
      const bool overlapped = false);
  void
  writeSolutionToMeshDatabase(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const double /* time */,
      const bool overlapped = false);
  void
  writeSolutionMVToMeshDatabase(
      const Thyra_MultiVector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double /* time */,
      const bool overlapped = false);

  //! Write the solution to file. Must call writeSolution first.
  void
  writeSolutionToFile(
      const Thyra_Vector& solution,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false); 
  void
  writeSolutionMVToFile(
      const Thyra_MultiVector& solution,
      const double             time,
      const bool               overlapped = false,
      const bool               force_write_solution = false); 

   /** Add a solution field
     */
   void addSolutionField(const std::string & fieldName,const std::string & blockId);

   /** Add a solution field
     */
   void addCellField(const std::string & fieldName,const std::string & blockId);

   //! get the dimension
   unsigned getDimension() const
   { return getNumDim(); }

   //! get the number of equations
   unsigned getNumberEquations() const
   { return neq; }

  //! used when NetCDF output on a latitude-longitude grid is requested.
  // Each struct contains a latitude/longitude index and it's parametric
  // coordinates in an element.
  struct interp
  {
    std::pair<double, double>     parametric_coords;
    std::pair<unsigned, unsigned> latitude_longitude;
  };

  void setFieldData(
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const Teuchos::RCP<StateInfoStruct>& sis);

  Teuchos::RCP<AbstractSTKFieldContainer> getSolutionFieldContainer() {
    return solutionFieldContainer;
  }

 protected:

  friend class BlockedSTKDiscretization;
  friend class STKConnManager;

  void
  getSolutionField(Thyra_Vector& result, bool overlapped) const;
  void
  getSolutionMV(Thyra_MultiVector& result, bool overlapped) const;

  void
  setSolutionField(const Thyra_Vector& soln, const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp, const bool overlapped);
  void
  setSolutionField(
      const Thyra_Vector& soln,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& soln_dot,
      const bool          overlapped);
  void
  setSolutionField(
      const Thyra_Vector& soln,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& soln_dot,
      const Thyra_Vector& soln_dotdot,
      const bool          overlapped);
  void
  setSolutionFieldMV(const Thyra_MultiVector& solnT,
		  const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
		  const bool overlapped);

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

  //! Find the local side id number within parent element
  unsigned
  determine_local_side_id(const stk::mesh::Entity elem, stk::mesh::Entity side);

  void
  writeCoordsToMatrixMarket() const;

  void
  buildSideSetProjectors();

  double previous_time_label;

  // ==================== Members =================== //

  Teuchos::RCP<Teuchos::FancyOStream> out;

  //! Stk Mesh Objects
  Teuchos::RCP<stk::mesh::MetaData> metaData;
  Teuchos::RCP<stk::mesh::BulkData> bulkData;

  //! Teuchos communicator
  Teuchos::RCP<const Teuchos_Comm> comm;

  //! Unknown map and node map
  Teuchos::RCP<const Thyra_VectorSpace> m_vs;
  Teuchos::RCP<const Thyra_VectorSpace> m_node_vs;

  //! Overlapped unknown map and node map
  Teuchos::RCP<const Thyra_VectorSpace> m_overlap_vs;
  Teuchos::RCP<const Thyra_VectorSpace> m_overlap_node_vs;

  //! Jacobian matrix operator factory
  Teuchos::RCP<ThyraCrsMatrixFactory> m_jac_factory;

  NodalDOFsStructContainer nodalDOFsStructContainer;

  //! Number of equations (and unknowns) per node
  const unsigned int neq;

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
  std::map<std::string, Kokkos::View<LO****, PHX::Device>> allLocalDOFViews;
  std::map<int, std::map<std::string, Kokkos::View<LO****, PHX::Device>>> wsLocalDOFViews;

  //! Connectivity array [workset, element, local-node, Eq] => LID
  Conn wsElNodeEqID;

  //! Connectivity array [workset, element, local-node] => GID
  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>        wsElNodeID;

  mutable Teuchos::ArrayRCP<double>                                 coordinates;
  Teuchos::RCP<Thyra_MultiVector>                                   coordMV;
  WorksetArray<std::string>                                         wsEBNames;
  WorksetArray<int>                                                 wsPhysIndex;
  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>       coords;
  WorksetArray<Teuchos::ArrayRCP<double>>        sphereVolume;
  WorksetArray<Teuchos::ArrayRCP<double*>>       latticeOrientation;

  //! Connectivity map from elementGID to workset and LID in workset
  WsLIDList elemGIDws;

  // States: vector of length worksets of a map from field name to shards array
  StateArrays                                   stateArrays;
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
  std::map<std::string, Teuchos::RCP<AbstractDiscretization>>
      sideSetDiscretizations;
  std::map<std::string, Teuchos::RCP<STKDiscretization>>
                                                        sideSetDiscretizationsSTK;
  std::map<std::string, std::map<GO, GO>>               sideToSideSetCellMap;
  std::map<std::string, std::map<GO, std::vector<int>>> sideNodeNumerationMap;
  std::map<std::string, Teuchos::RCP<Thyra_LinearOp>>   projectors;
  std::map<std::string, Teuchos::RCP<Thyra_LinearOp>>   ov_projectors;

// Used in Exodus writing capability
#ifdef ALBANY_SEACAS
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data;

  int outputInterval;

  size_t outputFileIdx;
#endif
  DiscType interleavedOrdering;

  Teuchos::RCP<AbstractSTKFieldContainer> solutionFieldContainer;
};

}  // namespace Albany

#endif  // ALBANY_STK_DISCRETIZATION_HPP
