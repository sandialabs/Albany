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

  static const char* solution_dof_name () { return "ordinary_solution"; }
  static const char* nodes_dof_name    () { return "mesh_nodes"; }

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
  virtual Teuchos::RCP<Thyra_LinearOp>
  createJacobianOp() const = 0;

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
  virtual const SideSetList&
  getSideSets(const int ws) const = 0;

  //! Get Side set view lists
  virtual const LocalSideSetInfoList&
  getSideSetViews(const int ws) const = 0;

  //! Get local DOF views for GatherVerticallyContractedSolution
  virtual const std::map<std::string, Kokkos::DualView<LO****, PHX::Device>>&
  getLocalDOFViews(const int workset) const = 0;

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
  virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>>>&
  getCoords() const = 0;

  //! Get coordinates (overlap map).
  virtual const Teuchos::ArrayRCP<double>&
  getCoordinates() const = 0;

  //! Print the coords for mesh debugging
  virtual void
  printCoords() const = 0;

  //! Get sideSet discretizations map
  const strmap_t<Teuchos::RCP<AbstractDiscretization>>& getSideSetDiscretizations() const { return sideSetDiscretizations; }

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

  //! Get stateArray of given type
  StateArrayVec& getStateArrays(const StateStruct::StateType type) {
    if (type==StateStruct::ElemState) {
      return getStateArrays().elemStateArrays;
    } else {
      return getStateArrays().nodeStateArrays;
    }
  }

  //! Get nodal parameters state info struct
  virtual const StateInfoStruct&
  getNodalParameterSIS() const = 0;

  //! Retrieve Vector (length num worksets) of element block names
  virtual const WorksetArray<std::string>&
  getWsEBNames() const = 0;

  //! Retrieve Vector (length num worksets) of Physics Index
  virtual const WorksetArray<int>&
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

  virtual Teuchos::RCP<Thyra_MultiVector>
  getSolutionMV(bool overlapped = false) const = 0;

  virtual void
  getField(Thyra_Vector& field_vector, const std::string& field_name) const = 0;
  virtual void
  setField(
      const Thyra_Vector& field_vector,
      const std::string&  field_name,
      bool                overlapped) = 0;

  virtual void
  setFieldData(const Teuchos::RCP<StateInfoStruct>& sis) = 0;

  // --- Methods to write solution in the output file --- //

  //! Write the solution to the output file. Calls next two together.
  virtual void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false) = 0; 
  virtual void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false) = 0; 
  virtual void
  writeSolution(
      const Thyra_Vector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const Thyra_Vector& solution_dot,
      const Thyra_Vector& solution_dotdot,
      const double        time,
      const bool          overlapped = false,
      const bool          force_write_solution = false) = 0; 
  virtual void
  writeSolutionMV(
      const Thyra_MultiVector& solution,
      const Teuchos::RCP<const Thyra_MultiVector>& solution_dxdp,
      const double             time,
      const bool               overlapped = false,
      const bool               force_write_solution = false) = 0; 
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
      const bool          overlapped = false,
      const bool          force_write_solution = false) = 0; 
  virtual void
  writeSolutionMVToFile(
      const Thyra_MultiVector& solution,
      const double             time,
      const bool               overlapped = false,
      const bool               force_write_solution = false) = 0; 

protected:
  strmap_t<Teuchos::RCP<AbstractDiscretization>> sideSetDiscretizations;

  // One dof mgr per dof per part
  // Notice that the dof mgr on a side is not the restriction
  // of the volume dof mgr to that side, since local ids are different.
  strmap_t<strmap_t<dof_mgr_ptr_t>>     m_dof_managers;

  // Dof manager for a scalar node field
  strmap_t<dof_mgr_ptr_t>               m_node_dof_managers;

  // The size of each workset
  WorksetArray<int>   m_workset_sizes;

  // For each workset, the element LID of its elements.
  // Note: with 1 workset, m_workset_elements(0,i)=i.
  DualView<int**>     m_workset_elements;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_DISCRETIZATION_HPP
