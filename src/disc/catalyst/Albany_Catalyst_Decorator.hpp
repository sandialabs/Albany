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

  //! Get overlapped DOF map
  Teuchos::RCP<const Epetra_Map> getOverlapMap() const override;

  //! Get Jacobian graph
  Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const override;

  //! Get overlap Jacobian graph
  Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const override;

  //! Get Node map
  Teuchos::RCP<const Epetra_Map> getNodeMap() const override;

  //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
  const NodeSetList& getNodeSets() const override;
  const NodeSetCoordList& getNodeSetCoords() const override;

  //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
  const SideSetList& getSideSets(const int workset) const override;

  //! Get map from (Ws, El, Local Node) -> NodeLID
  using AbstractDiscretization::Conn;
  const Conn& getWsElNodeEqID() const override;

  //! Retrieve coodinate vector (num_used_nodes * 3)
  const Teuchos::ArrayRCP<double>& getCoordinates() const override;
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const override;

  //! Print the coordinates for debugging
  void printCoords() const override;

  Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() const override;

  Albany::StateArrays& getStateArrays() override;

  //! Retrieve Vector (length num worksets) of element block names
  const WorksetArray<std::string>::type&  getWsEBNames() const override;

  //! Retrieve Vector (length num worksets) of physics set index
  const WorksetArray<int>::type&  getWsPhysIndex() const override;

  //! Get connectivity map from elementGID to workset
  WsLIDList& getElemGIDws() override;

  void writeSolution(const Epetra_Vector& soln, const double time, const bool overlapped = false) override;

  vtkUnstructuredGridBase* newVtkUnstructuredGrid();

  Teuchos::RCP<Epetra_Vector> getSolutionField(bool overlapped=false) const override;

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const override;

  virtual bool supportsMOR() const override;

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const override;

  //! Get number of spatial dimensions
  int getNumDim() const override;

  //! Get number of total DOFs per node
  int getNumEq() const override;

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
