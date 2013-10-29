//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBPUMI_FMDBDISCRETIZATION_HPP
#define ALBPUMI_FMDBDISCRETIZATION_HPP

#include <vector>
#include <fstream>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Epetra_Comm.h"

#include "AlbPUMI_AbstractPUMIDiscretization.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"
#include "AlbPUMI_FMDBVtk.hpp"
#include "AlbPUMI_FMDBExodus.hpp"

#include "Piro_NullSpaceUtils.hpp" // has defn of struct that holds null space info for ML

#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"

namespace AlbPUMI {

template<class Output>
  class FMDBDiscretization : public AbstractPUMIDiscretization {
  public:

    //! Constructor
    FMDBDiscretization(
       Teuchos::RCP<AlbPUMI::FMDBMeshStruct> fmdbMeshStruct,
       const Teuchos::RCP<const Epetra_Comm>& comm,
       const Teuchos::RCP<Piro::MLRigidBodyModes>& rigidBodyModes = Teuchos::null);


    //! Destructor
    ~FMDBDiscretization();

    //! Get DOF map
    Teuchos::RCP<const Epetra_Map> getMap() const;
    //! Get Tpetra DOF map
    Teuchos::RCP<const Tpetra_Map> getMapT() const;

    //! Get overlapped DOF map
    Teuchos::RCP<const Epetra_Map> getOverlapMap() const;
    //! Get Tpetra overlapped DOF map
    Teuchos::RCP<const Tpetra_Map> getOverlapMapT() const;

    //! Get Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const;
    //! Get Tpetra Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getJacobianGraphT() const;

    //! Get overlap Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const;
    //! Get Tpetra overlap Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getOverlapJacobianGraphT() const;

    //! Get Node map
    Teuchos::RCP<const Epetra_Map> getNodeMap() const;
    //! Get Tpetra Node map
    Teuchos::RCP<const Tpetra_Map> getNodeMapT() const;

    //! Get Overlap Node map
//    Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const;

    //! Process coords for ML
    void setupMLCoords();

    //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
    const Albany::NodeSetList& getNodeSets() const { return nodeSets; };
    const Albany::NodeSetCoordList& getNodeSetCoords() const { return nodeSetCoords; };

    //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
    const Albany::SideSetList& getSideSets(const int workset) const { return sideSets[workset]; };

   //! Get connectivity map from elementGID to workset
    Albany::WsLIDList& getElemGIDws() { return elemGIDws; };

    //! Get map from (Ws, El, Local Node) -> NodeLID
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type& getWsElNodeEqID() const;

    //! Retrieve coodinate vector (num_used_nodes * 3)
    Teuchos::ArrayRCP<double>& getCoordinates() const;

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const;

    // FIXME - Dummy FELIX accessor functions
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type& getSurfaceHeight() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type& getTemperature() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type& getBasalFriction() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type& getThickness() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type& getFlowFactor() const;

    //! Print coords for debugging
    void printCoords() const;
    void debugMeshWrite(const Epetra_Vector& sol, const char* filename);

   //! Get number of spatial dimensions
    int getNumDim() const { return fmdbMeshStruct->numDim; }

    virtual Teuchos::RCP<const Epetra_Comm> getComm() const { return comm; }

    //! Get number of total DOFs per node
    int getNumEq() const { return neq; }

    Albany::StateArrays& getStateArrays() {return stateArrays;};

    //! Retrieve Vector (length num worksets) of element block names
    const Albany::WorksetArray<std::string>::type&  getWsEBNames() const;
    //! Retrieve Vector (length num worksets) of physics set index
    const Albany::WorksetArray<int>::type&  getWsPhysIndex() const;

    //
    void writeSolution(const Epetra_Vector& soln, const double time, const bool overlapped = false);
    
    //Tpetra version of writeSolution 
    void writeSolutionT(const Tpetra_Vector& soln, const double time, const bool overlapped = false);

    Teuchos::RCP<Epetra_Vector> getSolutionField() const;
    //Tpetra analog
    Teuchos::RCP<Tpetra_Vector> getSolutionFieldT() const;

    void setResidualField(const Epetra_Vector& residual);
    //Tpetra analog
    void setResidualFieldT(const Tpetra_Vector& residualT);

    // Retrieve mesh struct
    Teuchos::RCP<AlbPUMI::FMDBMeshStruct> getFMDBMeshStruct() {return fmdbMeshStruct;}

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return fmdbMeshStruct->hasRestartSolution;}

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return fmdbMeshStruct->restartDataTime;}

    //! FMDB does not support MOR
    virtual bool supportsMOR() const { return false; }

    // After mesh modification, need to update the element connectivity and nodal coordinates
    void updateMesh();

    //! Accessor function to get coordinates for ML. Memory controlled here.
    void getOwned_xyz(double **x, double **y, double **z, double **rbm,
                      int& nNodes, int numPDEs, int numScalar, int nullSpaceDim);

    // Function that transforms an FMDB mesh of a unit cube (for FELIX problems)
    // not supported in FMDB now
    void transformMesh(){}

    inline int getOwnedDOF(const int inode, const int eq) const
    {
      if (interleavedOrdering) return inode*neq + eq;
      else  return inode + numOwnedNodes*eq;
    }

    inline int getOverlapDOF(const int inode, const int eq) const
    {
      if (interleavedOrdering) return inode*neq + eq;
      else  return inode + numOverlapNodes*eq;
    }

    inline int getGlobalDOF(const int inode, const int eq) const
    {
      if (interleavedOrdering) return inode*neq + eq;
      else  return inode + numGlobalNodes*eq;
    }

  private:

    //! Private to prohibit copying
    FMDBDiscretization(const FMDBDiscretization&);

    //! Private to prohibit copying
    FMDBDiscretization& operator=(const FMDBDiscretization&);

    // Copy solution vector from Epetra_Vector into FMDB Mesh
    // Here soln is the local (non overlapped) solution
    void setSolutionField(const Epetra_Vector& soln);

    // Copy solution vector from Epetra_Vector into FMDB Mesh
    // Here soln is the local + neighbor (overlapped) solution
    void setOvlpSolutionField(const Epetra_Vector& soln);

    int nonzeroesPerRow(const int neq) const;
    double monotonicTimeLabel(const double time);

    //! Process FMDB mesh for Owned nodal quantitites
    void computeOwnedNodesAndUnknowns();
    //! Process FMDB mesh for Overlap nodal quantitites
    void computeOverlapNodesAndUnknowns();
    //! Process FMDB mesh for CRS Graphs
    void computeGraphs();
    //! Process FMDB mesh for Workset/Bucket Info
    void computeWorksetInfo();
    //! Process FMDB mesh for NodeSets
    void computeNodeSets();
    //! Process FMDB mesh for SideSets
    void computeSideSets();
    //! Find the local side id number within parent element
//    unsigned determine_local_side_id( const stk::mesh::Entity & elem , stk::mesh::Entity & side );
    //! Call stk_io for creating exodus output file
    Teuchos::RCP<Teuchos::FancyOStream> out;

    double previous_time_label;

    // Transformation types for FELIX problems
    enum TRANSFORMTYPE {NONE, ISMIP_HOM_TEST_A};
    TRANSFORMTYPE transform_type;

  protected:

    //! Output object
    Output meshOutput;

    //! Stk Mesh Objects

    //! Epetra communicator
    Teuchos::RCP<const Epetra_Comm> comm;

   //! Tpetra communicator and Kokkos node
    Teuchos::RCP<const Teuchos::Comm<int> > commT;
    Teuchos::RCP<KokkosNode> nodeT;

    //! Node map
    Teuchos::RCP<const Tpetra_Map> node_mapT;

    //! Unknown Map
    Teuchos::RCP<const Tpetra_Map> mapT;

    //! Overlapped unknown map, and node map
    Teuchos::RCP<const Tpetra_Map> overlap_mapT;
    Teuchos::RCP<const Tpetra_Map> overlap_node_mapT;

    //! Jacobian matrix graph
    Teuchos::RCP<Tpetra_CrsGraph> graphT;

    //! Overlapped Jacobian matrix graph
    Teuchos::RCP<Tpetra_CrsGraph> overlap_graphT;

    //! Processor ID
    unsigned int myPID;

    //! Number of equations (and unknowns) per node
    const unsigned int neq;

    //! Number of elements on this processor
    unsigned int numMyElements;

    //! node sets stored as std::map(string ID, int vector of GIDs)
    Albany::NodeSetList nodeSets;
    Albany::NodeSetCoordList nodeSetCoords;

    //! side sets stored as std::map(string ID, SideArray classes) per workset (std::vector across worksets)
    std::vector<Albany::SideSetList> sideSets;

    //! Connectivity array [workset, element, local-node, Eq] => LID
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type wsElNodeEqID;

    mutable Teuchos::ArrayRCP<double> coordinates;
    Albany::WorksetArray<std::string>::type wsEBNames;
    Albany::WorksetArray<int>::type wsPhysIndex;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type coords;

    // FELIX unused variables (FIXME)
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type sHeight;
    Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type temperature;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type basalFriction;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type thickness;
    Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type flowFactor;

    //! Connectivity map from elementGID to workset and LID in workset
    Albany::WsLIDList  elemGIDws;

    // States: vector of length num worksets of a map from field name to shards array
    Albany::StateArrays stateArrays;

    //! Number of elements on this processor
    int numOwnedNodes;
    int numOverlapNodes;
    int numGlobalNodes;

    // Coordinate vector in format needed by ML. Need to own memory here.
    double *xx, *yy, *zz, *rr;
    bool allocated_xyz;

    // Storage used in periodic BCs to un-roll coordinates. Pointers saved for destructor.
    std::vector<double*>  toDelete;

    Teuchos::RCP<AlbPUMI::FMDBMeshStruct> fmdbMeshStruct;

    bool interleavedOrdering;

    std::vector< std::vector<pMeshEnt> > buckets; // bucket of elements

    // storage to save the node coordinates of the nodesets visible to this PE
    std::map<std::string, std::vector<double> > nodeset_node_coords;

    // Needed to pass coordinates to ML. 
    Teuchos::RCP<Piro::MLRigidBodyModes> rigidBodyModes;

    // counter for limiting data writes to output file
    int outputInterval;

  };

}

// Define macro for explicit template instantiation
#define FMDB_INSTANTIATE_TEMPLATE_CLASS_VTK(name) \
  template class name<AlbPUMI::FMDBVtk>;
#define FMDB_INSTANTIATE_TEMPLATE_CLASS_EXODUS(name) \
  template class name<AlbPUMI::FMDBExodus>;

#define FMDB_INSTANTIATE_TEMPLATE_CLASS(name) \
  FMDB_INSTANTIATE_TEMPLATE_CLASS_VTK(name) \
  FMDB_INSTANTIATE_TEMPLATE_CLASS_EXODUS(name)

#endif // ALBANY_FMDBDISCRETIZATION_HPP
