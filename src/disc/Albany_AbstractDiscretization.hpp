//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra ifdef'ed out if ALBANY_EPETRA_EXE turned off.

#ifndef ALBANY_ABSTRACTDISCRETIZATION_HPP
#define ALBANY_ABSTRACTDISCRETIZATION_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"

#if defined(ALBANY_EPETRA)
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsGraph.h"
#endif

#include "Shards_CellTopologyData.h"
#include "Shards_Array.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_NodalDOFManager.hpp"
#include "Albany_AbstractMeshStruct.hpp"

namespace AAdapt { namespace rc { class Manager; } }

namespace Albany {

typedef std::map<std::string, std::vector<std::vector<int> > > NodeSetList;
typedef std::map<std::string, std::vector<GO> > NodeSetGIDsList;
typedef std::map<std::string, std::vector<double*> > NodeSetCoordList;

class SideStruct {

  public:

    GO elem_GID; // the global id of the element containing the side
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

typedef std::map<GO, wsLid > WsLIDList;

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

#if defined(ALBANY_EPETRA)
    //! Get Epetra DOF map
    virtual Teuchos::RCP<const Epetra_Map> getMap() const = 0;
    //! Get field DOF map
    virtual Teuchos::RCP<const Epetra_Map> getMap(const std::string& field_name) const = 0;
#endif
    //! Get Tpetra DOF map
    virtual Teuchos::RCP<const Tpetra_Map> getMapT() const = 0;
    //! Get Tpetra DOF map
    //dp-convert virtual Teuchos::RCP<const Tpetra_Map> getMapT(const std::string& field_name) const = 0;

#if defined(ALBANY_EPETRA)
    //! Get Epetra overlapped DOF map
    virtual Teuchos::RCP<const Epetra_Map> getOverlapMap() const = 0;
    //! Get field overlapped DOF map
    virtual Teuchos::RCP<const Epetra_Map> getOverlapMap(const std::string& field_name) const = 0;
#endif
    //! Get Tpetra overlapped DOF map
    virtual Teuchos::RCP<const Tpetra_Map> getOverlapMapT() const = 0;
    //! Get field overlapped DOF map
    //virtual Teuchos::RCP<const Tpetra_Map> getOverlapMapT(const std::string& field_name) const = 0;

#if defined(ALBANY_EPETRA)
    //! Get Epetra Jacobian graph
    virtual Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const = 0;
#endif
    //! Get Tpetra Jacobian graph
    virtual Teuchos::RCP<const Tpetra_CrsGraph> getJacobianGraphT() const = 0;
   
#ifdef ALBANY_AERAS 
    //! Get implicit Tpetra Jacobian graph (for Aeras hyperviscosity)
    virtual Teuchos::RCP<const Tpetra_CrsGraph> getImplicitJacobianGraphT() const = 0;
#endif
    
    //! Get Epetra overlap Jacobian graph
#if defined(ALBANY_EPETRA)
    virtual Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const = 0;
#endif
    //! Get Tpetra overlap Jacobian graph
    virtual Teuchos::RCP<const Tpetra_CrsGraph> getOverlapJacobianGraphT() const = 0;
#ifdef ALBANY_AERAS 
    //! Get implicit Tpetra Jacobian graph (for Aeras hyperviscosity)
    virtual Teuchos::RCP<const Tpetra_CrsGraph> getImplicitOverlapJacobianGraphT() const = 0;
#endif

#if defined(ALBANY_EPETRA)
    //! Get Epetra Node map
    virtual Teuchos::RCP<const Epetra_Map> getNodeMap() const = 0;

    //! Get Field Node map
    virtual Teuchos::RCP<const Epetra_Map> getNodeMap(const std::string& field_name) const = 0;
#endif
    //! Get Tpetra Node map
    virtual Teuchos::RCP<const Tpetra_Map> getNodeMapT() const = 0;

#if defined(ALBANY_EPETRA)
    //! Get overlapped Node map
    virtual Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const = 0;

    //! Get Field overlapped Node map
    virtual Teuchos::RCP<const Epetra_Map> getOverlapNodeMap(const std::string& field_name) const = 0;
#endif
    //! Get overlapped Node map
    virtual Teuchos::RCP<const Tpetra_Map> getOverlapNodeMapT() const = 0;

    //! Get Node set lists
    virtual const NodeSetList& getNodeSets() const = 0;
    virtual const NodeSetGIDsList& getNodeSetGIDs() const = 0;
    virtual const NodeSetCoordList& getNodeSetCoords() const = 0;

    //! Get Side set lists
    virtual const SideSetList& getSideSets(const int ws) const = 0;

    //! Get map from (Ws, El, Local Node, Eq) -> unkLID
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type&
      getWsElNodeEqID() const = 0;

    //! Get map from (Ws, El, Local Node) -> unkGID
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
      getWsElNodeID() const = 0;

#if defined(ALBANY_EPETRA)
    //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for both scalar and vector fields
    virtual const std::vector<IDArray>& getElNodeEqID(const std::string& field_name) const = 0;

    //! Get Dof Manager of field field_name
    virtual const NodalDOFManager& getDOFManager(const std::string& field_name) const = 0;

    //! Get Dof Manager of field field_name
    virtual const NodalDOFManager& getOverlapDOFManager(const std::string& field_name) const = 0;
#endif

    //! Retrieve coodinate ptr_field (ws, el, node)
    virtual const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const = 0;

    //! Get coordinates (overlap map).
    virtual const Teuchos::ArrayRCP<double>& getCoordinates() const = 0;
    //! Set coordinates (overlap map) for mesh adaptation.
    virtual void setCoordinates(const Teuchos::ArrayRCP<const double>& c) = 0;

    //! The reference configuration manager handles updating the reference
    //! configuration. This is only relevant, and also only optional, in the
    //! case of mesh adaptation.
    virtual void setReferenceConfigurationManager(const Teuchos::RCP<AAdapt::rc::Manager>& rcm) = 0;

    virtual const WorksetArray<Teuchos::ArrayRCP<double> >::type& getSphereVolume() const = 0;

    //! Print the coords for mesh debugging
    virtual void printCoords() const = 0;

    //! Get stateArrays
    virtual Albany::StateArrays& getStateArrays() = 0;

    //! Get nodal parameters state info struct
    virtual const Albany::StateInfoStruct& getNodalParameterSIS() const = 0;

    //! Retrieve Vector (length num worksets) of element block names
    virtual const WorksetArray<std::string>::type& getWsEBNames() const = 0;

    //! Retrieve Vector (length num worksets) of Physics Index
    virtual const WorksetArray<int>::type& getWsPhysIndex() const = 0;

    //! Retrieve connectivity map from elementGID to workset
    virtual WsLIDList&  getElemGIDws() = 0;

#if defined(ALBANY_EPETRA)
    //! Get solution vector from mesh database
    virtual Teuchos::RCP<Epetra_Vector> getSolutionField(bool overlapped=false) const = 0;
#endif
    virtual Teuchos::RCP<Tpetra_Vector> getSolutionFieldT(bool overlapped=false) const = 0;

#if defined(ALBANY_EPETRA)
    //! Get field vector from mesh database
    virtual void getField(Epetra_Vector &field_vector, const std::string& field_name) const = 0;
#endif
    //dp-convert virtual void getFieldT(Tpetra_Vector &field_vector, const std::string& field_name) const = 0;

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

#if defined(ALBANY_EPETRA)
    //! Set the solution field into mesh database
    virtual void setSolutionField(const Epetra_Vector& soln) = 0;

    //! Set the field vector into mesh database
    virtual void setField(const Epetra_Vector &field_vector, const std::string& field_name, bool overlapped) = 0;

    //! Set the residual field for output
    virtual void setResidualField(const Epetra_Vector& residual) = 0;
#endif
    //! Set the residual field for output - Tpetra version
    virtual void setResidualFieldT(const Tpetra_Vector& residual) = 0;

#if defined(ALBANY_EPETRA)
    //! Write the solution to the output file
    virtual void writeSolution(const Epetra_Vector& solution, const double time, const bool overlapped = false) = 0;
#endif

    //! Write the solution to the output file - Tpetra version. Calls next two together.
    virtual void writeSolutionT(const Tpetra_Vector &solutionT, const double time, const bool overlapped = false) = 0;
    //! Write the solution to the mesh database.
    virtual void writeSolutionToMeshDatabaseT(const Tpetra_Vector &solutionT, const double time, const bool overlapped = false) = 0;
    //! Write the solution to file. Must call writeSolutionT first.
    virtual void writeSolutionToFileT(const Tpetra_Vector &solutionT, const double time, const bool overlapped = false) = 0;
    
    //! update the mesh
    virtual void updateMesh(bool shouldTransferIPData = false) = 0;

    //! Get Numbering for layered mesh (mesh structred in one direction)
    virtual Teuchos::RCP<LayeredMeshNumbering<LO> > getLayeredMeshNumbering() = 0;


  private:

    //! Private to prohibit copying
    AbstractDiscretization(const AbstractDiscretization&);

    //! Private to prohibit copying
    AbstractDiscretization& operator=(const AbstractDiscretization&);

};

}

#endif // ALBANY_ABSTRACTDISCRETIZATION_HPP
