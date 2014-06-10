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
#include "Albany_AbstractMeshStruct.hpp"
#include "Albany_StateInfoStruct.hpp"
//#include "Adapt_NodalDataBlock.hpp"

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

template <typename T>
struct WorksetArray {
   typedef Teuchos::ArrayRCP<T> type;
};

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

    //! Get Nodal block data
//    virtual Teuchos::RCP<Adapt::NodalDataBlock> getNodalDataBlock() = 0;

    //! Get Node set lists (typdef in Albany_Discretization.hpp)
    virtual const NodeSetList& getNodeSets() const = 0;
    virtual const NodeSetCoordList& getNodeSetCoords() const = 0;

    //! Get Side set lists
    virtual const SideSetList& getSideSets(const int ws) const = 0;

    //! Get map from (Ws, El, Local Node, Eq) -> unkLID
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
       getWsElNodeEqID() const = 0;

    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type&
       getWsElNodeID() const = 0;

    //! Retrieve coodinate ptr_field (ws, el, node)
    virtual Teuchos::ArrayRCP<double>&  getCoordinates() const = 0;
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const = 0;
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type& getSurfaceHeight() const = 0;
    virtual const WorksetArray<Teuchos::ArrayRCP<double> >::type& getTemperature() const = 0;
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type& getBasalFriction() const = 0;
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type& getThickness() const = 0;
    virtual const WorksetArray<Teuchos::ArrayRCP<double> >::type& getFlowFactor() const = 0;
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getSurfaceVelocity() const = 0;
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getVelocityRMS() const = 0;
    virtual const WorksetArray<Teuchos::ArrayRCP<double> >::type& getSphereVolume() const = 0;

    //! Print the coords for mesh debugging
    virtual void printCoords() const = 0;

    virtual Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() const = 0;

    virtual Albany::StateArrays& getStateArrays() = 0;

    //! Retrieve Vector (length num worksets) of element block names
    virtual const WorksetArray<std::string>::type&  getWsEBNames() const = 0;

    //! Retrieve Vector (length num worksets) of Physics Index
    virtual const WorksetArray<int>::type&  getWsPhysIndex() const = 0;

    //! Retrieve connectivity map from elementGID to workset
    virtual WsLIDList&  getElemGIDws() = 0;

    //! Get solution vector from mesh database
    virtual Teuchos::RCP<Epetra_Vector> getSolutionField() const = 0;

    //! Flag if solution has a restart values -- used in Init Cond
    virtual bool hasRestartSolution() const = 0;

    //! Does the underlying discretization support MOR?
    virtual bool supportsMOR() const = 0;

    //! File time of restart solution
    virtual double restartDataTime() const = 0;

    //! Get number of spatial dimensions
    virtual int getNumDim() const = 0;

    //! Get number of total DOFs per node
    virtual int getNumEq() const = 0;

    virtual void setSolutionField(const Epetra_Vector& soln){};

    //! Set the residual field for output
    virtual void setResidualField(const Epetra_Vector& residual) = 0;

    //! Write the solution to the output file
    virtual void writeSolution(const Epetra_Vector& solution, const double time, const bool overlapped = false) = 0;


  private:

    //! Private to prohibit copying
    AbstractDiscretization(const AbstractDiscretization&);

    //! Private to prohibit copying
    AbstractDiscretization& operator=(const AbstractDiscretization&);

};

}

#endif // ALBANY_ABSTRACTDISCRETIZATION_HPP
