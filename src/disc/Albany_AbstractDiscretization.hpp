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
#include "Albany_StateInfoStruct.hpp"
#include "Albany_DOFManager.hpp"
#include "Albany_ThyraCrsMatrixFactory.hpp"

#include "Albany_ThyraTypes.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_DualView.hpp"

namespace Albany {

class AbstractDiscretization
{
public:
  template<typename T>
  using strmap_t = std::map<std::string,T>;

  using conn_mgr_ptr_t = Teuchos::RCP<Albany::ConnManager>;
  using dof_mgr_ptr_t  = Teuchos::RCP<Albany::DOFManager>;

  static std::string solution_dof_name () { return "ordinary_solution"; }
  static std::string nodes_dof_name    () { return "mesh_nodes"; }

  //! Constructor
  AbstractDiscretization() = default;

  //! Prohibit copying
  AbstractDiscretization(const AbstractDiscretization&) = delete;
  AbstractDiscretization&
  operator=(const AbstractDiscretization&) = default;

  //! Destructor
  virtual ~AbstractDiscretization() = default;

  //! Get the DOF manager
  Teuchos::RCP<const DOFManager>
  getDOFManager (const std::string& fieldName) const {
    TEUCHOS_TEST_FOR_EXCEPTION (m_dof_managers.find(fieldName)==m_dof_managers.end(), std::runtime_error,
        "Error! Could not find a dof manger for field '" + fieldName + "'\n");
    TEUCHOS_TEST_FOR_EXCEPTION (m_dof_managers.at(fieldName).size()!=1, std::runtime_error,
        "Error! Multiple dof mangers for field '" + fieldName + "', and no part name specified.\n");

    return m_dof_managers.at(fieldName).begin()->second;
  }
  Teuchos::RCP<const DOFManager>
  getDOFManager (const std::string& fieldName, const std::string& part_name) const {
    return m_dof_managers.at(fieldName).at(part_name);
  }

  Teuchos::RCP<const DOFManager>
  getNodeDOFManager (const std::string& part_name) const {
    return m_node_dof_managers.at(part_name);
  }

  Teuchos::RCP<const DOFManager>
  getDOFManager () const
  {
    return getDOFManager (solution_dof_name(), "");
  }

  Teuchos::RCP<const DOFManager>
  getNodeDOFManager () const
  {
    return getNodeDOFManager("");
  }

  // Check if a dof manager for a particular field on a particular part exists
  bool hasDOFManager (const std::string& field_name, const std::string& part_name) const {
    return m_dof_managers.count(field_name)>0 &&
           m_dof_managers.at(field_name).count(part_name)>0;
  }

  //! Get node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace>
  getNodeVectorSpace() const
  {
    return getNodeDOFManager()->vs();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapNodeVectorSpace() const
  {
    return getNodeDOFManager()->ov_vs();
  }

  //! Get solution DOF vector space (owned and overlapped).
  Teuchos::RCP<const Thyra_VectorSpace>
  getVectorSpace() const
  {
    return getDOFManager()->vs();
  }

  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapVectorSpace() const
  {
    return getDOFManager()->ov_vs();
  }

  //! Get Field node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace>
  getNodeVectorSpace(const std::string& field_name) const
  {
    auto part_name = getDOFManager(field_name)->part_name();
    return getNodeDOFManager(part_name)->vs();
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapNodeVectorSpace(const std::string& field_name) const
  {
    auto part_name = getDOFManager(field_name)->part_name();
    return getNodeDOFManager(part_name)->ov_vs();
  }

  //! Get Field vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace>
  getVectorSpace(const std::string& field_name) const
  {
    return getDOFManager(field_name)->vs();
  }
  Teuchos::RCP<const Thyra_VectorSpace>
  getOverlapVectorSpace(const std::string& field_name)
  {
    return getDOFManager(field_name)->ov_vs();
  }

  //! Create a Jacobian operator
  Teuchos::RCP<Thyra_LinearOp> createJacobianOp() const
  {
    return m_jac_factory->createOp();
  }

  //! Get Node set lists
  virtual const NodeSetList&
  getNodeSets() const = 0;
  virtual const NodeSetGIDsList&
  getNodeSetGIDs() const = 0;
  virtual const NodeSetCoordList&
  getNodeSetCoords() const = 0;

  const WorksetArray<int>& getWorksetsSizes () const { return m_workset_sizes; }

  DualView<const int**>
  getWsElementLIDs () const { return m_workset_elements; }

  DualView<const int*>::host_t
  getElementLIDs_host (const int ws) const {
    constexpr auto ALL = Kokkos::ALL();
    return Kokkos::subview (m_workset_elements.host(),ws,ALL);
  }

  int getNumWorksets () const { return m_workset_sizes.size(); }

  //! Get Side set lists
  const SideSetList& getSideSets(const int ws) const { return m_sideSets[ws]; }

  //! Get Side set view lists
  const LocalSideSetInfoList&
  getSideSetViews(const int ws) const { return m_sideSetViews.at(ws); }

  //! Get local DOF views for GatherVerticallyContractedSolution
  const std::map<std::string, Kokkos::DualView<LO****, PHX::Device>>&
  getLocalDOFViews(const int workset) const
  {
    return m_wsLocalDOFViews.at(workset);
  }

  //! Get Dof Manager of field field_name
  Teuchos::RCP<const GlobalLocalIndexer>
  getGlobalLocalIndexer(const std::string& field_name) const
  {
    return getDOFManager(field_name)->indexer();
  }

  //! Get Dof Manager of field field_name
  Teuchos::RCP<const GlobalLocalIndexer>
  getOverlapGlobalLocalIndexer(const std::string& field_name) const
  {
    return getDOFManager(field_name)->ov_indexer();
  }

  //! Get GlobalLocalIndexer for solution field
  Teuchos::RCP<const GlobalLocalIndexer>
  getGlobalLocalIndexer () const { return getGlobalLocalIndexer(solution_dof_name()); }

  Teuchos::RCP<const GlobalLocalIndexer>
  getSidesGlobalLocalIndexer() const { return m_sides_indexer; }

  Teuchos::RCP<const GlobalLocalIndexer>
  getCellsGlobalLocalIndexer() const { return getDOFManager()->cell_indexer(); }

  //! Get GlobalLocalIndexer for overlapped solution field
  Teuchos::RCP<const GlobalLocalIndexer>
  getOverlapGlobalLocalIndexer () const { return getOverlapGlobalLocalIndexer(solution_dof_name()); }

  //! Get GlobalLocalIndexer for node field
  Teuchos::RCP<const GlobalLocalIndexer>
  getNodeGlobalLocalIndexer () const { return getGlobalLocalIndexer(nodes_dof_name()); }

  //! Get GlobalLocalIndexer for overlapped node field
  Teuchos::RCP<const GlobalLocalIndexer>
  getOverlapNodeGlobalLocalIndexer () const { return getOverlapGlobalLocalIndexer(nodes_dof_name()); }

  //! Retrieve coodinate ptr_field (ws, el, node)
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<const double*>>>&
  getCoords() const { return m_ws_elem_coords; }

  //! Get coordinates (overlap map).
  virtual const Teuchos::ArrayRCP<double>&
  getCoordinates() const = 0;

  //! Print the coords for mesh debugging
  virtual void
  printCoords() const = 0;

  //! Get sideSet discretizations map
  const strmap_t<Teuchos::RCP<AbstractDiscretization>>& getSideSetDiscretizations() const { return sideSetDiscretizations; }

  //! Get the map side_id->side_set_elem_id
  const strmap_t<std::map<GO, GO>>& getSideToSideSetCellMap() const {
    return m_side_to_ss_cell;
  }

  //! Get the map side_node_id->side_set_cell_node_id
  const strmap_t<std::map<GO, std::vector<int>>>& getSideNodeNumerationMap() const {
    return m_side_nodes_to_ss_cell_nodes;
  }

  //! Get MeshStruct
  virtual Teuchos::RCP<AbstractMeshStruct>
  getMeshStruct() const = 0;

  //! Get nodal parameters state info struct
  const StateInfoStruct& getNodalParameterSIS() const {
    return getMeshStruct()->get_field_accessor()->getNodalParameterSIS();
  }

  //! Retrieve Vector (length num worksets) of element block names
  const WorksetArray<std::string>&
  getWsEBNames() const { return m_wsEBNames; }

  //! Retrieve Vector (length num worksets) of Physics Index
  const WorksetArray<int>&
  getWsPhysIndex() const { return m_wsPhysIndex; }

  //! Retrieve array storing the ws idx and the idx within the ws of each element (indexed via elem LID)
  const std::vector<WsIdx>& get_elements_workset_idx () const { return m_elem_ws_idx; }
        std::vector<WsIdx>& get_elements_workset_idx ()       { return m_elem_ws_idx; }

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
  int getNumEq() const { return m_neq; }

  //! Get Numbering for layered mesh (mesh structured in one direction)
  virtual Teuchos::RCP<LayeredMeshNumbering<GO>>
  getLayeredMeshNumberingGO() const {
    return getMeshStruct()->global_cell_layers_data;
  }
  virtual Teuchos::RCP<LayeredMeshNumbering<LO>>
  getLayeredMeshNumberingLO() const {
    return getMeshStruct()->local_cell_layers_data;
  }

  // --- Get/set solution/residual/field vectors to/from mesh --- //
  virtual Teuchos::RCP<Thyra_Vector>
  getSolutionField(bool overlapped = false) const = 0;

  virtual void
  getSolutionMV(Thyra_MultiVector& soln, bool overlapped = false) const = 0;

  virtual void
  getSolutionDxDp(Thyra_MultiVector& dxdp, bool overlapped = false) const = 0;

  virtual void
  getField(Thyra_Vector& field_vector, const std::string& field_name) const = 0;
  virtual void
  setField(
      const Thyra_Vector& field_vector,
      const std::string&  field_name,
      bool                overlapped) = 0;

  virtual void setFieldData() = 0;

  // Update mesh internals, such as coordinates, DOF numbers, etc.
  // To be run either after creation or after modification/adaptation.
  virtual void updateMesh () {};

  // --- Methods to write solution in the output file --- //

  //! All these overloads call a corresponding overload of writeSolutionToMeshDatabase and writeMeshDatabaseToFile
  void writeSolution(const Thyra_Vector& solution,
                     const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                     const double        time,
                     const bool          overlapped = false,
                     const bool          force_write_solution = false);
  void writeSolution (const Thyra_Vector& solution,
                      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                      const Thyra_Vector& solution_dot,
                      const double        time,
                      const bool          overlapped = false,
                      const bool          force_write_solution = false);
  void writeSolution (const Thyra_Vector& solution,
                      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                      const Thyra_Vector& solution_dot,
                      const Thyra_Vector& solution_dotdot,
                      const double        time,
                      const bool          overlapped = false,
                      const bool          force_write_solution = false);
  virtual void writeSolutionMV (const Thyra_MultiVector& solution,
                                const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                const double             time,
                                const bool               overlapped = false,
                                const bool               force_write_solution = false);

  //! Write the solution to the mesh database.
  virtual void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                            const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                            const bool          overlapped) = 0;
  virtual void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                            const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                            const Thyra_Vector& solution_dot,
                                            const bool          overlapped) = 0;
  virtual void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                            const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                            const Thyra_Vector& solution_dot,
                                            const Thyra_Vector& solution_dotdot,
                                            const bool          overlapped) = 0;
  virtual void writeSolutionMVToMeshDatabase (const Thyra_MultiVector& solution,
                                              const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
                                              const bool               overlapped) = 0;

  //! Write the solution to file. Must call writeSolution first.
  virtual void writeMeshDatabaseToFile (const double time,
                                        const bool   force_write_solution) = 0;

  // Check if mesh adaptation is needed, and if so what kind (topological or just mesh-movement)
  virtual Teuchos::RCP<AdaptationData>
  checkForAdaptation (const Teuchos::RCP<const Thyra_Vector>& solution,
                      const Teuchos::RCP<const Thyra_Vector>& solution_dot,
                      const Teuchos::RCP<const Thyra_Vector>& solution_dotdot,
                      const Teuchos::RCP<const Thyra_MultiVector>& dxdp) = 0;

  // Check if mesh adaptation is needed, and if so adapt mesh (and possibly reinterpolate solution)
  virtual void adapt (const Teuchos::RCP<AdaptationData>& adaptData) = 0;

  virtual Teuchos::RCP<ConnManager>
  create_conn_mgr (const std::string& part_name) {
    throw std::runtime_error("Error! This discretization does not implement 'create_conn_mgr'.");
  }

protected:
  dof_mgr_ptr_t&
  get_dof_mgr (const std::string& part_name,
               const FE_Type fe_type,
               const int order,
               const int dof_dim);

  // From std::vector<SideSet> build corresponding kokkos structures
  void buildSideSetsViews ();

  strmap_t<Teuchos::RCP<AbstractDiscretization>> sideSetDiscretizations;

  //! Jacobian matrix operator factory
  Teuchos::RCP<ThyraCrsMatrixFactory> m_jac_factory;

  // Notice that the dof mgr on a side is not the restriction
  // of the volume dof mgr to that side, since local ids are different.
  // Note: the double map works as map[field_name][part_name] = dof_mgr
  strmap_t<strmap_t<dof_mgr_ptr_t>>     m_dof_managers;

  // Dof manager for a scalar node field (part_name->dof_mgr)
  strmap_t<dof_mgr_ptr_t>               m_node_dof_managers;

  // Store a all dof mgrs based on a key that encodes all params used to create it.
  // This helps to build only one copy of dof mgrs with same specs
  std::map<std::string,dof_mgr_ptr_t>      m_key_to_dof_mgr;

  // For each ss, map the side_GID into vec, where vec[i]=k if the i-th
  // node of the side corresponds to the k-th node of the cell in the side mesh
  strmap_t<std::map<GO, std::vector<int>>> m_side_nodes_to_ss_cell_nodes;

  // For each ss, map the side_GID to the cell_GID in the side mesh
  strmap_t<std::map<GO, GO>>               m_side_to_ss_cell;

  //! side sets stored as std::map(string ID, SideArray classes) per workset
  //! (std::vector across worksets)
  std::vector<SideSetList>            m_sideSets;
  std::map<int, LocalSideSetInfoList> m_sideSetViews;

  // Provide side gid<->lid indexing. GIDs and LIDs are unique across worksets and sidesets
  Teuchos::RCP<const GlobalLocalIndexer>    m_sides_indexer;

  // The index to which each element belongs to (indexed by elem LID)
  std::vector<WsIdx>  m_elem_ws_idx;

  //! Number of equations (and unknowns) per node
  // TODO: this should soon be removed, in favor of more granular description of each dof/unknown
  int m_neq;

  //! GatherVerticallyContractedSolution connectivity
  std::map<int, std::map<std::string, Kokkos::DualView<LO****, PHX::Device>>> m_wsLocalDOFViews;

  // Workset information
  WorksetArray<int>           m_workset_sizes; // size of each ws
  WorksetArray<std::string>   m_wsEBNames;     // name of elem block that ws belongs
  WorksetArray<int>           m_wsPhysIndex;   // physics index of each ws

  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<const double*>>> m_ws_elem_coords;

  // For each workset, the element LID of its elements.
  // Note: with 1 workset, m_workset_elements(0,i)=i.
  DualView<int**>     m_workset_elements;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_DISCRETIZATION_HPP
