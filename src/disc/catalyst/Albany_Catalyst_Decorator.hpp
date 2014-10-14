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
  Teuchos::RCP<const Epetra_Map> getMap() const;

  //! Get overlapped DOF map
  Teuchos::RCP<const Epetra_Map> getOverlapMap() const;

  //! Get Jacobian graph
  Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const;

  //! Get overlap Jacobian graph
  Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const;

  //! Get Node map
  Teuchos::RCP<const Epetra_Map> getNodeMap() const;

  //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
  const NodeSetList& getNodeSets() const;
  const NodeSetCoordList& getNodeSetCoords() const;

  //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
  const SideSetList& getSideSets(const int workset) const;

  //! Get map from (Ws, El, Local Node) -> NodeLID
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type& getWsElNodeEqID() const;

  //! Retrieve coodinate vector (num_used_nodes * 3)
  Teuchos::ArrayRCP<double>& getCoordinates() const;
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const;

  //! Print the coordinates for debugging
  void printCoords() const;

  Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() const;

  Albany::StateArrays& getStateArrays();

  //! Retrieve Vector (length num worksets) of element block names
  const WorksetArray<std::string>::type&  getWsEBNames() const;

  //! Retrieve Vector (length num worksets) of physics set index
  const WorksetArray<int>::type&  getWsPhysIndex() const;

  //! Get connectivity map from elementGID to workset
  WsLIDList& getElemGIDws();

  void writeSolution(const Epetra_Vector& soln, const double time, const bool overlapped = false);

  vtkUnstructuredGridBase* newVtkUnstructuredGrid();

  Teuchos::RCP<Epetra_Vector> getSolutionField() const;

  void setResidualField(const Epetra_Vector& residual);

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const;

  virtual bool supportsMOR() const;

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const;

  //! Get number of spatial dimensions
  int getNumDim() const;

  //! Get number of total DOFs per node
  int getNumEq() const;

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
