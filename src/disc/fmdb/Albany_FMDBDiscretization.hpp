//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_FMDBDISCRETIZATION_HPP
#define ALBANY_FMDBDISCRETIZATION_HPP

#include <vector>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Epetra_Comm.h"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_FMDBMeshStruct.hpp"

#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"

namespace Albany {

  class FMDBDiscretization : public Albany::AbstractDiscretization {
  public:

    //! Constructor
    FMDBDiscretization(
       Teuchos::RCP<Albany::FMDBMeshStruct> fmdbMeshStruct,
       const Teuchos::RCP<const Epetra_Comm>& comm);


    //! Destructor
    ~FMDBDiscretization();

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
    const NodeSetList& getNodeSets() const { return nodeSets; };
    const NodeSetCoordList& getNodeSetCoords() const { return nodeSetCoords; };

    //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
    const SideSetList& getSideSets(const int workset) const { return sideSets[workset]; };

   //! Get connectivity map from elementGID to workset
    WsLIDList& getElemGIDws() { return elemGIDws; };

    //! Get map from (Ws, El, Local Node) -> NodeLID
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >& getWsElNodeEqID() const;

    //! Retrieve coodinate vector (num_used_nodes * 3)
    Teuchos::ArrayRCP<double>& getCoordinates() const;

    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >& getCoords() const;

    Albany::StateArrays& getStateArrays() {return stateArrays;};

    //! Retrieve Vector (length num worksets) of element block names
    const Teuchos::ArrayRCP<std::string>&  getWsEBNames() const;
    //! Retrieve Vector (length num worksets) of physics set index
    const Teuchos::ArrayRCP<int>&  getWsPhysIndex() const;

    // 
    void outputToExodus(const Epetra_Vector& soln, const double time, const bool overlapped = false);
 
    Teuchos::RCP<Epetra_Vector> getSolutionField() const;

    void setResidualField(const Epetra_Vector& residual);

    // Retrieve mesh struct
    Teuchos::RCP<Albany::FMDBMeshStruct> getFMDBMeshStruct() {return fmdbMeshStruct;};

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return fmdbMeshStruct->hasRestartSolution;}

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return fmdbMeshStruct->restartDataTime;}

    // After mesh modification, need to update the element connectivity and nodal coordinates
    void updateMesh(Teuchos::RCP<Albany::FMDBMeshStruct> fmdbMeshStruct,
    		const Teuchos::RCP<const Epetra_Comm>& comm);

    //! Accessor function to get coordinates for ML. Memory controlled here.
    void getOwned_xyz(double **x, double **y, double **z, double **rbm,
                      int& nNodes, int numPDEs, int numScalar, int nullSpaceDim);

    // Function that transforms an FMDB mesh of a unit cube (for FELIX problems)
    // not supported
    void transformMesh(){}

    void freeAllocatedMemory();

  private:

    //! Private to prohibit copying
    FMDBDiscretization(const FMDBDiscretization&);

    //! Private to prohibit copying
    FMDBDiscretization& operator=(const FMDBDiscretization&);

    // dof calc  nodeID*neq+eqID
//    inline int gid(const stk::mesh::Entity& node) const;
//    inline int gid(const stk::mesh::Entity* node) const;

    inline int getOwnedDOF(const int inode, const int eq) const;
    inline int getOverlapDOF(const int inode, const int eq) const;
    inline int getGlobalDOF(const int inode, const int eq) const;

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
    //! Call stk_io for creating exodus output file
//    void setupExodusOutput();
    //! Find the local side id number within parent element
//    unsigned determine_local_side_id( const stk::mesh::Entity & elem , stk::mesh::Entity & side );
    //! Call stk_io for creating exodus output file
    Teuchos::RCP<Teuchos::FancyOStream> out;

    double previous_time_label;

    // Transformation types for FELIX problems
    enum TRANSFORMTYPE {NONE, ISMIP_HOM_TEST_A};
    TRANSFORMTYPE transform_type;

  protected:

    
    //! Stk Mesh Objects

    //! Epetra communicator
    Teuchos::RCP<const Epetra_Comm> comm;

    //! Node map
    Teuchos::RCP<Epetra_Map> node_map;

    //! Unknown Map
    Teuchos::RCP<Epetra_Map> map;

    //! Overlapped unknown map, and node map
    Teuchos::RCP<Epetra_Map> overlap_map;
    Teuchos::RCP<Epetra_Map> overlap_node_map;

    //! Jacobian matrix graph
    Teuchos::RCP<Epetra_CrsGraph> graph;

    //! Overlapped Jacobian matrix graph
    Teuchos::RCP<Epetra_CrsGraph> overlap_graph;

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
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > > wsElNodeEqID;

    mutable Teuchos::ArrayRCP<double> coordinates;
    Teuchos::ArrayRCP<std::string> wsEBNames;
    Teuchos::ArrayRCP<int> wsPhysIndex;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords;

    //! Connectivity map from elementGID to workset and LID in workset
    WsLIDList  elemGIDws;

    // States: vector of length num worksets of a map from field name to shards array
    Albany::StateArrays stateArrays;

    //! list of all owned nodes, saved for setting solution
//    std::vector< stk::mesh::Entity * > ownednodes ;
//    std::vector< stk::mesh::Entity * > cells ;

    //! list of all overlap nodes, saved for getting coordinates for mesh motion
//    std::vector< stk::mesh::Entity * > overlapnodes ;

    //! Number of elements on this processor
    int numOwnedNodes;
    int numOverlapNodes;
    int numGlobalNodes;

    // Coordinate vector in format needed by ML. Need to own memory here.
    double *xx, *yy, *zz, *rr;
    bool allocated_xyz;

    // Storage used in periodic BCs to un-roll coordinates. Pointers saved for destructor.
    std::vector<double*>  toDelete;

    Teuchos::RCP<Albany::FMDBMeshStruct> fmdbMeshStruct;

    bool interleavedOrdering;

//    std::vector< std::vector<pMeshEnt> > buckets; // bucket of elements
    std::vector< std::vector<pMeshEnt>* > buckets; // bucket of elements
  };

}

#endif // ALBANY_FMDBDISCRETIZATION_HPP
