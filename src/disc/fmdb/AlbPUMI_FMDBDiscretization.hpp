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

    //! Get overlapped DOF map
    Teuchos::RCP<const Epetra_Map> getOverlapMap() const;

    //! Get Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const;

    //! Get overlap Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const;

    //! Get Node map
    Teuchos::RCP<const Epetra_Map> getNodeMap() const;

    //! Get Overlap Node map
    Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const;

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

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type& getWsElNodeID() const;

    //! Retrieve coodinate vector (num_used_nodes * 3)
    Teuchos::ArrayRCP<double>& getCoordinates() const;

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const;

    // FIXME - Dummy FELIX accessor functions
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type& getSurfaceHeight() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type& getTemperature() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type& getBasalFriction() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type& getThickness() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getSurfaceVelocity() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getVelocityRMS() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type& getSphereVolume() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type& getFlowFactor() const;

    //! Print coords for debugging
    void printCoords() const;
    void debugMeshWriteNative(const Epetra_Vector& sol, const char* filename);
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

    void writeSolution(const Epetra_Vector& soln, const double time, const bool overlapped = false);

    Teuchos::RCP<Epetra_Vector> getSolutionField() const;

    void setResidualField(const Epetra_Vector& residual);

    // Retrieve mesh struct
    Teuchos::RCP<AlbPUMI::FMDBMeshStruct> getFMDBMeshStruct() {return fmdbMeshStruct;}
    Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() const {return fmdbMeshStruct;}

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return fmdbMeshStruct->hasRestartSolution;}

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return fmdbMeshStruct->restartDataTime;}

    //! FMDB does not support MOR
    virtual bool supportsMOR() const { return false; }

    // Before mesh modification, qp data may needed for solution transfer
    void attachQPData();

    // After mesh modification, qp data needs to be detached
    void detachQPData();

    // After mesh modification, need to update the element connectivity and nodal coordinates
    void updateMesh(bool shouldTransferIPData);

    //! Accessor function to get coordinates for ML. Memory controlled here.
    void getOwned_xyz(double **x, double **y, double **z, double **rbm,
                      int& nNodes, int numPDEs, int numScalar, int nullSpaceDim);

    // Function that transforms an FMDB mesh of a unit cube (for FELIX problems)
    // not supported in FMDB now
    void transformMesh(){}

    int getDOF(const int inode, const int eq) const
    {
      if (interleavedOrdering) return inode*neq + eq;
      else  return inode + numOwnedNodes*eq;
    }

    int getOwnedDOF(const int inode, const int eq) const
    {
      return getDOF(inode,eq);
    }

    int getOverlapDOF(const int inode, const int eq) const
    {
      return getDOF(inode,eq);
    }

    int getGlobalDOF(const int inode, const int eq) const
    {
      return getDOF(inode,eq);
    }

    // Copy field data from Epetra_Vector to APF
    void setField(
        const char* name,
        const Epetra_Vector& data,
        bool overlapped,
        int offset = 0);
    void setSplitFields(std::vector<std::string> names, std::vector<int> indices, 
        const Epetra_Vector& data, bool overlapped);

    // Copy field data from APF to Epetra_Vector
    void getField(
        const char* name,
        Epetra_Vector& data,
        bool overlapped,
        int offset = 0) const;
    void getSplitFields(std::vector<std::string> names, std::vector<int> indices,
        Epetra_Vector& data, bool overlapped) const;

    // Rename exodus output file when the problem is resized
    void reNameExodusOutput(const std::string& str){ meshOutput.setFileName(str);}

  private:

    //! Private to prohibit copying
    FMDBDiscretization(const FMDBDiscretization&);

    //! Private to prohibit copying
    FMDBDiscretization& operator=(const FMDBDiscretization&);

    // Copy solution vector from Epetra_Vector into FMDB Mesh
    // Here soln is the local (non overlapped) solution
    void setSolutionField(const Epetra_Vector& soln);

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

    //! Transfer QPData to APF
    void copyQPScalarToAPF(unsigned nqp, QPData<double, 2>& state, apf::Field* f);
    void copyQPVectorToAPF(unsigned nqp, QPData<double, 3>& state, apf::Field* f);
    void copyQPTensorToAPF(unsigned nqp, QPData<double, 4>& state, apf::Field* f);
    void copyQPStatesToAPF(apf::Field* f, apf::FieldShape* fs);
    void removeQPStatesFromAPF();

    //! Transfer QP Fields from APF to QPData
    void copyQPScalarFromAPF(unsigned nqp, QPData<double, 2>& state, apf::Field* f);
    void copyQPVectorFromAPF(unsigned nqp, QPData<double, 3>& state, apf::Field* f);
    void copyQPTensorFromAPF(unsigned nqp, QPData<double, 4>& state, apf::Field* f);
    void copyQPStatesFromAPF();

    // ! Split Solution fields
    std::vector<std::string> solNames;
    std::vector<std::string> resNames;
    std::vector<int> solIndex;

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

    //! Number of equations (and unknowns) per node
    const unsigned int neq;

    //! node sets stored as std::map(string ID, int vector of GIDs)
    Albany::NodeSetList nodeSets;
    Albany::NodeSetCoordList nodeSetCoords;

    //! side sets stored as std::map(string ID, SideArray classes) per workset (std::vector across worksets)
    std::vector<Albany::SideSetList> sideSets;

    //! Connectivity array [workset, element, local-node, Eq] => LID
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type wsElNodeEqID;

    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type wsElNodeID;

    mutable Teuchos::ArrayRCP<double> coordinates;
    Albany::WorksetArray<std::string>::type wsEBNames;
    Albany::WorksetArray<int>::type wsPhysIndex;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type coords;

    // FELIX unused variables (FIXME)
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type sHeight;
    Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type temperature;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type basalFriction;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type thickness;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type surfaceVelocity;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type velocityRMS;
    Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type flowFactor;
    Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type sphereVolume;

    //! Connectivity map from elementGID to workset and LID in workset
    Albany::WsLIDList  elemGIDws;

    // States: vector of length num worksets of a map from field name to shards array
    Albany::StateArrays stateArrays;

    //! list of all overlap nodes, saved for setting solution
    apf::DynamicArray<apf::Node> nodes;

    //! Number of elements on this processor
    int numOwnedNodes;
    int numOverlapNodes;
    int numGlobalNodes;

    // Coordinate vector in format needed by ML. Need to own memory here.
    double *xx, *yy, *zz, *rr;
    bool allocated_xyz;

    Teuchos::RCP<AlbPUMI::FMDBMeshStruct> fmdbMeshStruct;

    bool interleavedOrdering;

    std::vector< std::vector<apf::MeshEntity*> > buckets; // bucket of elements

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
