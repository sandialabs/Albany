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
#ifdef ALBANY_EPETRA
#include "Epetra_Comm.h"
#endif

#include "AlbPUMI_AbstractPUMIDiscretization.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"
#include "AlbPUMI_FMDBVtk.hpp"
#include "AlbPUMI_FMDBExodus.hpp"

#include "Albany_NullSpaceUtils.hpp"
#ifdef ALBANY_EPETRA
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"
#endif

namespace AlbPUMI {

template<class Output>
  class FMDBDiscretization : public AbstractPUMIDiscretization {
  public:

    //! Constructor
    FMDBDiscretization(
       Teuchos::RCP<AlbPUMI::FMDBMeshStruct> fmdbMeshStruct,
       const Teuchos::RCP<const Teuchos_Comm>& commT,
       const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes = Teuchos::null);


    //! Destructor
    ~FMDBDiscretization();

    //! Get Tpetra DOF map
    Teuchos::RCP<const Tpetra_Map> getMapT() const;

    //! Get Tpetra overlapped DOF map
    Teuchos::RCP<const Tpetra_Map> getOverlapMapT() const;

    //! Get Tpetra Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getJacobianGraphT() const;

    //! Get Tpetra overlap Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getOverlapJacobianGraphT() const;

    //! Get Tpetra Node map
    Teuchos::RCP<const Tpetra_Map> getNodeMapT() const;

    //! Get Tpetra overlap Node map
    Teuchos::RCP<const Tpetra_Map> getOverlapNodeMapT() const;

    //! Process coords for ML
    void setupMLCoords();

    //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
    const Albany::NodeSetList& getNodeSets() const { return nodeSets; };
    const Albany::NodeSetCoordList& getNodeSetCoords() const { return nodeSetCoords; };

    //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
    const Albany::SideSetList& getSideSets(const int workset) const { return sideSets[workset]; };

    //! Get connectivity map from elementGID to workset
    Albany::WsLIDList& getElemGIDws() { return elemGIDws; };

    //! Get map from (Ws, El, Local Node, Eqn) -> dof LID
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type&
    getWsElNodeEqID() const;

    //! Get map from (Ws, El, Local Node) -> NodeGID
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type& getWsElNodeID() const;

    //! Get coordinate vector (overlap map, interleaved)
    const Teuchos::ArrayRCP<double>& getCoordinates() const;
    //! Set coordinate vector (overlap map, interleaved)
    void setCoordinates(const Teuchos::ArrayRCP<const double>& c);
    void setReferenceConfigurationManager(const Teuchos::RCP<AAdapt::rc::Manager>& rcm);

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const;

    const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type& getSphereVolume() const;

    //! Print coords for debugging
    void printCoords() const;

    //! Get number of spatial dimensions
    int getNumDim() const { return fmdbMeshStruct->numDim; }

    virtual Teuchos::RCP<const Teuchos_Comm> getComm() const { return commT; }

    //! Get number of total DOFs per node
    int getNumEq() const { return neq; }

    Albany::StateArrays& getStateArrays() {return stateArrays;};

    //! Retrieve Vector (length num worksets) of element block names
    const Albany::WorksetArray<std::string>::type& getWsEBNames() const;
    //! Retrieve Vector (length num worksets) of physics set index
    const Albany::WorksetArray<int>::type&  getWsPhysIndex() const;

    void writeAnySolutionToMeshDatabase(const ST* soln, const double time, const bool overlapped = false);
    void writeAnySolutionToFile(const ST* soln, const double time, const bool overlapped = false);
    void writeSolutionT(const Tpetra_Vector& soln, const double time, const bool overlapped = false);
    void writeSolutionToMeshDatabaseT(const Tpetra_Vector& soln, const double time, const bool overlapped = false);
    void writeSolutionToFileT(const Tpetra_Vector& soln, const double time, const bool overlapped = false);

    virtual void writeMeshDebug (const std::string& filename);

    Teuchos::RCP<Tpetra_Vector> getSolutionFieldT(bool overlapped=false) const;

    void setResidualFieldT(const Tpetra_Vector& residualT);

    // Retrieve mesh struct
    Teuchos::RCP<AlbPUMI::FMDBMeshStruct> getFMDBMeshStruct() {return fmdbMeshStruct;}
    Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() const {return fmdbMeshStruct;}

    Teuchos::RCP<SideSetDiscretizations> getSideSetDiscretizations () const
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Functionality not supported in FMDB discretization.\n");
      return Teuchos::null;
    }

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return fmdbMeshStruct->hasRestartSolution;}

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return fmdbMeshStruct->restartDataTime;}

    //! FMDB does not support MOR
    virtual bool supportsMOR() const { return false; }

    apf::GlobalNumbering* getAPFGlobalNumbering() {return elementNumbering;}

    // Before mesh modification, qp data may be needed for solution transfer
    void attachQPData();

    // After mesh modification, qp data needs to be removed
    void detachQPData();

    // After mesh modification, need to update the element connectivity and nodal coordinates
    void updateMesh(bool shouldTransferIPData);

    //! Accessor function to get coordinates for ML. Memory controlled here.
    void getOwned_xyz(double **x, double **y, double **z, double **rbm,
                      int& nNodes, int numPDEs, int numScalar, int nullSpaceDim);

    // Function that transforms an FMDB mesh of a unit cube (for FELIX problems)
    // not supported in FMDB now
    void transformMesh(){}

    // this is called with both LO's and GO's to compute a dof number
    // based on a node number and an equation number
    GO getDOF(const GO inode, const int entry, int nentries=-1) const
    {
      if (interleavedOrdering) {
        if (nentries == -1) nentries = neq;
        return inode*nentries + entry;
      }
      else return inode + numOwnedNodes*entry;
    }

    // Copy field data from Tpetra_Vector to APF
    void setField(const char* name, const ST* data, bool overlapped,
                  int offset = 0, int nentries = -1);
    void setSplitFields(std::vector<std::string> names, std::vector<int> indices,
        const ST* data, bool overlapped);

    // Copy field data from APF to Tpetra_Vector
    void getField(const char* name, ST* dataT, bool overlapped, int offset = 0,
                  int nentries = -1) const;
    void getSplitFields(std::vector<std::string> names, std::vector<int> indices,
        ST* dataT, bool overlapped) const;

    void createField(const char* name, int value_type);

    // Rename exodus output file when the problem is resized
    void reNameExodusOutput(const std::string& str){ meshOutput.setFileName(str);}

    /* DAI: old Epetra functions still used by parts of Albany/Trilinos
       Remove when we get to full Tpetra */
#ifdef ALBANY_EPETRA
    virtual Teuchos::RCP<const Epetra_Map> getMap() const { return map; }
    virtual Teuchos::RCP<const Epetra_Map> getOverlapMap() const { return overlap_map; }
    virtual Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const;
    virtual Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const { return graph; }
    virtual Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const { return overlap_graph; }
    virtual Teuchos::RCP<const Epetra_Map> getNodeMap() const {
      fprintf(stderr,"PUMI Discretization unsupported call getNodeMap\n");
      abort();
      return Teuchos::RCP<const Epetra_Map>();
    }
    virtual Teuchos::RCP<Epetra_Vector> getSolutionField(bool overlapped=false) const;
    virtual void setResidualField(const Epetra_Vector& residual);
    virtual void writeSolution(const Epetra_Vector&, const double, const bool);
    void setSolutionField(const Epetra_Vector&) {
      fprintf(stderr,"PUMI Discretization unsupported call setSolutionField\n");
      abort();
    }
    void debugMeshWriteNative(const Epetra_Vector&, const char*) {
      fprintf(stderr,"PUMI Discretization unsupported call debugMeshWriteNative\n");
      abort();
    }
    void debugMeshWrite(const Epetra_Vector&, const char*) {
      fprintf(stderr,"PUMI Discretization unsupported call debugMeshWrite\n");
      abort();
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
    //! Get field DOF map
    Teuchos::RCP<const Epetra_Map> getMap(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "AlbPUMI:FMDBDiscretization: getMap(field_name) not implemented yet");
    }
    //! Get field overlapped DOF map
    Teuchos::RCP<const Epetra_Map> getOverlapMap(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "AlbPUMI:FMDBDiscretization: getOverlapMap(field_name) not implemented yet");
    }
    //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for both scalar and vector fields
    const std::vector<Albany::IDArray>& getElNodeEqID(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "AlbPUMI:FMDBDiscretization: getElNodeElID(field_name) not implemented yet");
    }
    //! Get field vector from mesh database
    virtual void getField(Epetra_Vector &field_vector, const std::string& field_name) const  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "AlbPUMI:FMDBDiscretization: getField(field_vector, field_name) not implemented yet");
    }
    //! Set the field vector into mesh database
    virtual void setField(const Epetra_Vector &field_vector, const std::string& field_name, bool overlapped)  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "AlbPUMI:FMDBDiscretization: setField(field_vector, field_name, overlapped) not implemented yet");
    }
#endif
    //! Get nodal parameters state info struct
    virtual const Albany::StateInfoStruct& getNodalParameterSIS() const  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "AlbPUMI:FMDBDiscretization: getNodalParameterSIS() not implemented yet");
    }

  private:

    //! Private to prohibit copying
    FMDBDiscretization(const FMDBDiscretization&);

    //! Private to prohibit copying
    FMDBDiscretization& operator=(const FMDBDiscretization&);

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
    void copyQPStatesToAPF(apf::Field* f, apf::FieldShape* fs, bool copyAll = true);
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

#ifdef ALBANY_EPETRA
    //! Epetra communicator
    Teuchos::RCP<const Epetra_Comm> comm;
#endif

   //! Tpetra communicator and Kokkos node
    Teuchos::RCP<const Teuchos::Comm<int> > commT;

    //! Node map
    Teuchos::RCP<const Tpetra_Map> node_mapT;

    //! Unknown Map
    Teuchos::RCP<const Tpetra_Map> mapT;
#ifdef ALBANY_EPETRA
    Teuchos::RCP<Epetra_Map> map;
#endif

    //! Overlapped unknown map, and node map
    Teuchos::RCP<const Tpetra_Map> overlap_mapT;
#ifdef ALBANY_EPETRA
    Teuchos::RCP<Epetra_Map> overlap_map;
#endif
    Teuchos::RCP<const Tpetra_Map> overlap_node_mapT;

    //! Jacobian matrix graph
    Teuchos::RCP<Tpetra_CrsGraph> graphT;
#ifdef ALBANY_EPETRA
    Teuchos::RCP<Epetra_CrsGraph> graph;
#endif

    //! Overlapped Jacobian matrix graph
    Teuchos::RCP<Tpetra_CrsGraph> overlap_graphT;
#ifdef ALBANY_EPETRA
    Teuchos::RCP<Epetra_CrsGraph> overlap_graph;
#endif

    //! Number of equations (and unknowns) per node
    const unsigned int neq;

    //! node sets stored as std::map(string ID, int vector of GIDs)
    Albany::NodeSetList nodeSets;
    Albany::NodeSetCoordList nodeSetCoords;

    //! side sets stored as std::map(string ID, SideArray classes) per workset (std::vector across worksets)
    std::vector<Albany::SideSetList> sideSets;

    //! Connectivity array [workset, element, local-node, Eq] => LID
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type wsElNodeEqID;

    //! Connectivity array [workset, element, local-node] => GID
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type wsElNodeID;

    mutable Teuchos::ArrayRCP<double> coordinates;
    Albany::WorksetArray<std::string>::type wsEBNames;
    Albany::WorksetArray<int>::type wsPhysIndex;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type coords;
    Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type sphereVolume;

    //! Connectivity map from elementGID to workset and LID in workset
    Albany::WsLIDList  elemGIDws;

    // States: vector of length num worksets of a map from field name to shards array
    Albany::StateArrays stateArrays;

    apf::GlobalNumbering* globalNumbering;
    apf::GlobalNumbering* elementNumbering;

    //! list of all overlap nodes, saved for setting solution
    apf::DynamicArray<apf::Node> nodes;

    //! Number of elements on this processor
    int numOwnedNodes;
    int numOverlapNodes;
    long numGlobalNodes;

    Teuchos::RCP<AlbPUMI::FMDBMeshStruct> fmdbMeshStruct;

    bool interleavedOrdering;

    std::vector< std::vector<apf::MeshEntity*> > buckets; // bucket of elements

    // storage to save the node coordinates of the nodesets visible to this PE
    std::map<std::string, std::vector<double> > nodeset_node_coords;

    // Needed to pass coordinates to ML.
    Teuchos::RCP<Albany::RigidBodyModes> rigidBodyModes;

    // counter for limiting data writes to output file
    int outputInterval;

    // Mesh adaptation stuff.
    Teuchos::RCP<AAdapt::rc::Manager> rcm;
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
