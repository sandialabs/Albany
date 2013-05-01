//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACTDISCRETIZATION_HPP
#define ALBANY_ABSTRACTDISCRETIZATION_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsGraph.h"

#include "Shards_CellTopologyData.h"
#include "Shards_Array.hpp"
#include "Albany_StateInfoStruct.hpp"

namespace Albany {

  typedef std::map<std::string, std::vector<std::vector<int> > > NodeSetList;
  typedef std::map<std::string, std::vector<double*> > NodeSetCoordList;

  class SideStruct {

    public:

    int elem_GID; // the global id of the element containing the side
    int elem_LID; // the local id of the element containing the side
    int elem_ebIndex; // The index of the element block that contains the element
    unsigned side_local_id; // The local id of the side relative to the owning element

  };

  typedef std::map<std::string, std::vector<SideStruct> > SideSetList;

  class wsLid {

    public:

    int ws; // the workset of the element containing the side
    int LID; // the local id of the element containing the side

  };

  typedef std::map<int, wsLid > WsLIDList;

  class AbstractDiscretization {
  public:

    //! Constructor
    AbstractDiscretization() {};

    //! Destructor
    virtual ~AbstractDiscretization() {};

    //! Get DOF map
    virtual Teuchos::RCP<const Epetra_Map>
    getMap() const = 0;

    //! Get overlapped DOF map
    virtual Teuchos::RCP<const Epetra_Map>
    getOverlapMap() const = 0;

    //! Get Jacobian graph
    virtual Teuchos::RCP<const Epetra_CrsGraph>
    getJacobianGraph() const = 0;

    //! Get overlap Jacobian graph
    virtual Teuchos::RCP<const Epetra_CrsGraph>
    getOverlapJacobianGraph() const = 0;

    //! Get Node map
    virtual Teuchos::RCP<const Epetra_Map>
    getNodeMap() const = 0;

    //! Get Node set lists (typdef in Albany_Discretization.hpp)
    virtual const NodeSetList& getNodeSets() const = 0;
    virtual const NodeSetCoordList& getNodeSetCoords() const = 0;

    //! Get Side set lists
    virtual const SideSetList& getSideSets(const int ws) const = 0;

    //! Get map from (Ws, El, Local Node, Eq) -> unkLID
    virtual const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >&
       getWsElNodeEqID() const = 0;

    //! Retrieve coodinate ptr_field (ws, el, node)
    virtual Teuchos::ArrayRCP<double>&  getCoordinates() const = 0;
    virtual const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >& getCoords() const = 0;
    virtual const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >& getSurfaceHeight() const = 0;

    virtual Albany::StateArrays& getStateArrays() = 0;

    //! Retrieve Vector (length num worksets) of element block names
    virtual const Teuchos::ArrayRCP<std::string>&  getWsEBNames() const = 0;

    //! Retrieve Vector (length num worksets) of Physics Index
    virtual const Teuchos::ArrayRCP<int>&  getWsPhysIndex() const = 0;

    //! Retrieve connectivity map from elementGID to workset
    virtual WsLIDList&  getElemGIDws() = 0;

    //! Get solution vector from mesh database
    virtual Teuchos::RCP<Epetra_Vector> getSolutionField() const = 0;

    //! Flag if solution has a restart values -- used in Init Cond
    virtual bool hasRestartSolution() const = 0;

    //! File time of restart solution
    virtual double restartDataTime() const = 0;

    //! Get number of spatial dimensions
    virtual int getNumDim() const = 0;

    //! Get number of total DOFs per node
    virtual int getNumEq() const = 0;

    //! Accessor function to get coordinates for ML. Memory controlled here.
    virtual void getOwned_xyz(double **x, double **y, double **z, double **rbm,
                              int& nNodes, int numPDEs, int numScalar, int nullSpaceDim) = 0;

  private:

    //! Private to prohibit copying
    AbstractDiscretization(const AbstractDiscretization&);

    //! Private to prohibit copying
    AbstractDiscretization& operator=(const AbstractDiscretization&);

  };

}

#endif // ALBANY_ABSTRACTDISCRETIZATION_HPP
