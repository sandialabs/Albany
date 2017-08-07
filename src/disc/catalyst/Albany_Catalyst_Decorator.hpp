//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_CATALYSTDECORATOR_HPP
#define ALBANY_CATALYSTDECORATOR_HPP

#include "Albany_AbstractDiscretization.hpp"

class vtkUnstructuredGridBase;

namespace Albany {
namespace Catalyst {

class Decorator : public AbstractDiscretization
{
public:
  Decorator(
      Teuchos::RCP<Albany::AbstractDiscretization> discretization_,
      const Teuchos::RCP<Teuchos::ParameterList>& catalystParams);
  ~Decorator();

  //! Get DOF map
  Teuchos::RCP<const Epetra_Map> getMap() const override;
  //! Get Tpetra DOF map
  Teuchos::RCP<const Tpetra_Map> getMapT() const override;
  //! Get Tpetra DOF map
  Teuchos::RCP<const Tpetra_Map> getMapT(const std::string& field_name) const override;

  //! Get overlapped DOF map
  Teuchos::RCP<const Epetra_Map> getOverlapMap() const override;
  //! Get Tpetra overlapped DOF map
  Teuchos::RCP<const Tpetra_Map> getOverlapMapT() const override;
  //! Get field overlapped DOF map
  Teuchos::RCP<const Tpetra_Map> getOverlapMapT(
      const std::string& field_name) const override;

  //! Get Jacobian graph
  Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const override;
  //! Get Tpetra Jacobian graph
  Teuchos::RCP<const Tpetra_CrsGraph> getJacobianGraphT() const override;

#ifdef ALBANY_AERAS 
  //! Get implicit Tpetra Jacobian graph (for Aeras hyperviscosity)
  Teuchos::RCP<const Tpetra_CrsGraph> getImplicitJacobianGraphT() const override;
#endif

  //! Get overlap Jacobian graph
  Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const override;
  //! Get Tpetra overlap Jacobian graph
  Teuchos::RCP<const Tpetra_CrsGraph> getOverlapJacobianGraphT() const override;
#ifdef ALBANY_AERAS 
  //! Get implicit Tpetra Jacobian graph (for Aeras hyperviscosity)
  Teuchos::RCP<const Tpetra_CrsGraph> getImplicitOverlapJacobianGraphT() const override;
#endif

  //! Get Node map
  Teuchos::RCP<const Epetra_Map> getNodeMap() const override;

  //! Get Tpetra Node map
  Teuchos::RCP<const Tpetra_Map> getNodeMapT() const override;

  //! Get Field Node map
  Teuchos::RCP<const Tpetra_Map> getNodeMapT(
      const std::string& field_name) const override;

#if defined(ALBANY_EPETRA)
  //! Get overlapped Node map
  Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const override;
#endif

  //! Get overlapped Node map
  Teuchos::RCP<const Tpetra_Map> getOverlapNodeMapT() const override;

  //! Returns boolean telling code whether explicit scheme is used (needed for Aeras problems only) 
  bool isExplicitScheme() const override;

  //! Get Field Node map
  Teuchos::RCP<const Tpetra_Map> getOverlapNodeMapT(
      const std::string& field_name) const override;

  //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
  const NodeSetList& getNodeSets() const override;
  const NodeSetGIDsList& getNodeSetGIDs() const override;
  const NodeSetCoordList& getNodeSetCoords() const override;

  //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
  const SideSetList& getSideSets(const int workset) const override;

  //! Get map from (Ws, El, Local Node) -> NodeLID
  using AbstractDiscretization::Conn;
  const Conn& getWsElNodeEqID() const override;

  //! Get map from (Ws, El, Local Node) -> unkGID
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    getWsElNodeID() const override;

  //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for both scalar and vector fields
  const std::vector<IDArray>& getElNodeEqID(const std::string& field_name) const override;

  //! Get Dof Manager of field field_name
  const NodalDOFManager& getDOFManager(
      const std::string& field_name) const override;

  //! Get Dof Manager of field field_name
  const NodalDOFManager& getOverlapDOFManager(
      const std::string& field_name) const override;

  //! Retrieve coodinate vector (num_used_nodes * 3)
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const override;

  //! Get coordinates (overlap map).
  const Teuchos::ArrayRCP<double>& getCoordinates() const override;
  //! Set coordinates (overlap map) for mesh adaptation.
  void setCoordinates(const Teuchos::ArrayRCP<const double>& c) override;

  //! The reference configuration manager handles updating the reference
  //! configuration. This is only relevant, and also only optional, in the
  //! case of mesh adaptation.
  void setReferenceConfigurationManager(
      const Teuchos::RCP<AAdapt::rc::Manager>& rcm) override;

#ifdef ALBANY_CONTACT
  //! Get the contact manager
  Teuchos::RCP<const Albany::ContactManager> getContactManager() const override;
#endif

  const WorksetArray<Teuchos::ArrayRCP<double> >::type& getSphereVolume() const override;

  const WorksetArray<Teuchos::ArrayRCP<double*> >::type& getLatticeOrientation() const override;

  //! Print the coordinates for debugging
  void printCoords() const override;

  //! Get sideSet discretizations map
  const SideSetDiscretizationsType& getSideSetDiscretizations() const override;

  //! Get the map side_id->side_set_elem_id
  const std::map<std::string,std::map<GO,GO>>& getSideToSideSetCellMap() const override;

  //! Get the map side_node_id->side_set_cell_node_id
  const std::map<std::string,std::map<GO,std::vector<int> > >& getSideNodeNumerationMap() const override;

  Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() const override;

  Albany::StateArrays& getStateArrays() override;

  //! Get nodal parameters state info struct
  const Albany::StateInfoStruct& getNodalParameterSIS() const override;

  //! Retrieve Vector (length num worksets) of element block names
  const WorksetArray<std::string>::type&  getWsEBNames() const override;

  //! Retrieve Vector (length num worksets) of physics set index
  const WorksetArray<int>::type&  getWsPhysIndex() const override;

  //! Get connectivity map from elementGID to workset
  WsLIDList& getElemGIDws() override;
  const WsLIDList&  getElemGIDws() const override;

  vtkUnstructuredGridBase* newVtkUnstructuredGrid();

  Teuchos::RCP<Epetra_Vector> getSolutionField(bool overlapped=false) const override;

  Teuchos::RCP<Tpetra_Vector> getSolutionFieldT(bool overlapped=false) const override;

  Teuchos::RCP<Tpetra_MultiVector> getSolutionMV(bool overlapped=false) const override;

  void getFieldT(Tpetra_Vector &field_vector, const std::string& field_name) const override;

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const override;

  bool supportsMOR() const override;

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const override;

  //! Get number of spatial dimensions
  int getNumDim() const override;

  //! Get number of total DOFs per node
  int getNumEq() const override;

  //! Set the field vector into mesh database
  void setFieldT(const Tpetra_Vector &field_vector, const std::string& field_name, bool overlapped) override;

  //! Set the residual field for output - Tpetra version
  void setResidualFieldT(const Tpetra_Vector& residual) override;

#if defined(ALBANY_EPETRA)
  void writeSolution(const Epetra_Vector& soln, const double time, const bool overlapped = false) override;
  void writeSolution(const Epetra_Vector& solution, const Epetra_Vector& solution_dot, 
                             const double time, const bool overlapped = false) override;
#endif

  //! Write the solution to the output file - Tpetra version. Calls next two together.
  void writeSolutionT(const Tpetra_Vector &solutionT, const double time, const bool overlapped = false) override;
  void writeSolutionT(const Tpetra_Vector &solutionT, const Tpetra_Vector &solution_dotT, 
                              const double time, const bool overlapped = false) override;
  void writeSolutionT(const Tpetra_Vector &solutionT, const Tpetra_Vector &solution_dotT, 
                              const Tpetra_Vector &solution_dotdotT, 
                              const double time, const bool overlapped = false) override;
  void writeSolutionMV(const Tpetra_MultiVector &solutionT, const double time, const bool overlapped = false) override;
  //! Write the solution to the mesh database.
  void writeSolutionToMeshDatabaseT(const Tpetra_Vector &solutionT, const double time, const bool overlapped = false) override;
  void writeSolutionToMeshDatabaseT(const Tpetra_Vector &solutionT, 
                                            const Tpetra_Vector &solution_dotT, 
                                            const double time, const bool overlapped = false) override;
  void writeSolutionToMeshDatabaseT(const Tpetra_Vector &solutionT, 
                                            const Tpetra_Vector &solution_dotT, 
                                            const Tpetra_Vector &solution_dotdotT, 
                                            const double time, const bool overlapped = false) override;
  void writeSolutionMVToMeshDatabase(const Tpetra_MultiVector &solutionT, const double time, const bool overlapped = false) override;
  //! Write the solution to file. Must call writeSolutionT first.
  void writeSolutionToFileT(const Tpetra_Vector &solutionT, const double time, const bool overlapped = false) override;
  void writeSolutionMVToFile(const Tpetra_MultiVector &solutionT, const double time, const bool overlapped = false) override;

  //! Get Numbering for layered mesh (mesh structred in one direction)
  Teuchos::RCP<LayeredMeshNumbering<LO> > getLayeredMeshNumbering() override;

private:
  //! Private to prohibit copying
  Decorator(const Decorator&);

  //! Private to prohibit copying
  Decorator& operator=(const Decorator&);

  Teuchos::RCP<Albany::AbstractDiscretization> discretization;
  int timestep;
};

} // end namespace Catalyst
} // end namespace Albany

#endif // ALBANY_CATALYSTDECORATOR_HPP
