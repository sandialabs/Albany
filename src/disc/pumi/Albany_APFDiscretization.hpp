//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_APFDISCRETIZATION_HPP
#define ALBANY_APFDISCRETIZATION_HPP

#include <vector>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Comm.h"
#endif

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_APFMeshStruct.hpp"
#include "Albany_PUMIOutput.hpp"

#include "Albany_NullSpaceUtils.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"
#endif

namespace Albany {

class APFDiscretization : public Albany::AbstractDiscretization {
  public:

    //! Constructor
    APFDiscretization(
       Teuchos::RCP<Albany::APFMeshStruct> meshStruct_in,
       const Teuchos::RCP<const Teuchos_Comm>& commT,
       const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes = Teuchos::null);

    //! Destructor
    virtual ~APFDiscretization();

    //! Initialize this class
    void init();

    //! Set any restart data
    virtual void setRestartData() {}

    //! Get Tpetra DOF map
    Teuchos::RCP<const Tpetra_Map> getMapT() const;

    //! Get Tpetra overlapped DOF map
    Teuchos::RCP<const Tpetra_Map> getOverlapMapT() const;

#ifdef ALBANY_AERAS
    // GAH - Is this right for Aeras? Probably do not use PUMI with Aeras I am assuming.
    Teuchos::RCP<const Tpetra_CrsGraph> getImplicitOverlapJacobianGraphT() const
                 { return this->getOverlapJacobianGraphT(); }
    Teuchos::RCP<const Tpetra_CrsGraph> getImplicitJacobianGraphT() const
                 { return this->getJacobianGraphT(); }
#endif

    //! Get Tpetra Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getJacobianGraphT() const;

    //! Get Tpetra overlap Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getOverlapJacobianGraphT() const;

    //! Get Tpetra Node map
    Teuchos::RCP<const Tpetra_Map> getNodeMapT() const;

    //! Get Tpetra overlap Node map
    Teuchos::RCP<const Tpetra_Map> getOverlapNodeMapT() const;

    virtual bool isExplicitScheme() const { return false; }

    //! Process coords for ML
    void setupMLCoords();

    //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
    const Albany::NodeSetList& getNodeSets() const { return nodeSets; };
    const Albany::NodeSetCoordList& getNodeSetCoords() const { return nodeSetCoords; };
    // not used; just completing concrete impl
    const Albany::NodeSetGIDsList& getNodeSetGIDs() const { return nodeSetGIDs; };

    //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
    const Albany::SideSetList& getSideSets(const int workset) const { return sideSets[workset]; };

    //! Get connectivity map from elementGID to workset
    Albany::WsLIDList& getElemGIDws() { return elemGIDws; };
    const Albany::WsLIDList& getElemGIDws() const { return elemGIDws; };

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

    const Albany::WorksetArray<Teuchos::ArrayRCP<double*> >::type& getLatticeOrientation() const;

    //! Print coords for debugging
    void printCoords() const;

    //! Get sideSet discretizations map
    const SideSetDiscretizationsType& getSideSetDiscretizations () const
    {
      //Warning: returning empty SideSetDiscretization for now.
      return sideSetDiscretizations;
    }

    //! Get the map side_id->side_set_elem_id
    const std::map<std::string,std::map<GO,GO> >& getSideToSideSetCellMap () const
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Functionality not supported by APF discretization.\n");
      return sideToSideSetCellMap;
    }

    //! Get the map side_node_id->side_set_cell_node_id
    const std::map<std::string,std::map<GO,std::vector<int>>>& getSideNodeNumerationMap () const
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Functionality not supported by APF discretization.\n");
      return sideNodeNumerationMap;
    }

    //! Get number of spatial dimensions
    int getNumDim() const { return meshStruct->numDim; }

    virtual Teuchos::RCP<const Teuchos_Comm> getComm() const { return commT; }

    //! Get number of total DOFs per node
    int getNumEq() const { return neq; }

    Albany::StateArrays& getStateArrays() {return stateArrays;};

    //! Retrieve Vector (length num worksets) of element block names
    const Albany::WorksetArray<std::string>::type& getWsEBNames() const;
    //! Retrieve Vector (length num worksets) of physics set index
    const Albany::WorksetArray<int>::type&  getWsPhysIndex() const;

    void writeAnySolutionToMeshDatabase(const ST* soln, const int index, const bool overlapped = false);
    void writeAnySolutionToFile(const double time);
    void writeSolutionT(const Tpetra_Vector& soln, const double time, const bool overlapped = false);
    void writeSolutionMV(const Tpetra_MultiVector& soln, const double time, const bool overlapped = false);
    void writeSolutionToMeshDatabaseT(const Tpetra_Vector& soln, const double time, const bool overlapped = false);
    void writeSolutionMVToMeshDatabase(const Tpetra_MultiVector& soln, const double time, const bool overlapped = false);
    void writeSolutionToFileT(const Tpetra_Vector& soln, const double time, const bool overlapped = false);
    void writeSolutionMVToFile(const Tpetra_MultiVector& soln, const double time, const bool overlapped = false);

    void writeRestartFile(const double time);

    virtual void writeMeshDebug (const std::string& filename);

    Teuchos::RCP<Tpetra_Vector> getSolutionFieldT(bool overlapped=false) const;

    Teuchos::RCP<Tpetra_MultiVector> getSolutionMV(bool overlapped=false) const;

    void setResidualFieldT(const Tpetra_Vector& residualT);

    // Retrieve mesh struct
    Teuchos::RCP<Albany::APFMeshStruct> getAPFMeshStruct() {return meshStruct;}
    Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() const {return meshStruct;}

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return meshStruct->hasRestartSolution;}

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return meshStruct->restartDataTime;}

    //! PUMI does not support MOR
    virtual bool supportsMOR() const { return false; }

    apf::GlobalNumbering* getAPFGlobalNumbering() {return elementNumbering;}

    // Before mesh modification, qp data may be needed for solution transfer
    void attachQPData();

    // After mesh modification, qp data needs to be removed
    void detachQPData();

    // After mesh modification, need to update the element connectivity and nodal coordinates
    void updateMesh(bool shouldTransferIPData);
    // The parameter library is used to update Time after adapting
    void updateMesh(bool shouldTransferIPData, Teuchos::RCP<ParamLib> paramLib);

    // Function that transforms a mesh of a unit cube (for FELIX problems)
    // not supported in PUMI now
    void transformMesh(){}

    // this is called with both LO's and GO's to compute a dof number
    // based on a node number and an equation number
    GO getDOF(const GO inode, const int entry, int total_comps = -1) const
    {
      if (interleavedOrdering) {
        if (total_comps == -1) total_comps = neq;
        return inode * total_comps + entry;
      }
      else return inode + numOwnedNodes*entry;
    }

    // Copy field data from Tpetra_Vector to APF
    void setField(const char* name, const ST* data, bool overlapped,
                  int offset = 0, bool neq_sized = true);
    void setSplitFields(const Teuchos::Array<std::string>& names,
                        const Teuchos::Array<int>& indices,
                        const ST* data, bool overlapped);

    // Copy field data from APF to Tpetra_Vector
    void getField(const char* name, ST* dataT, bool overlapped,
                  int offset = 0, bool neq_sized = true) const;
    void getSplitFields(const Teuchos::Array<std::string>& names,
                        const Teuchos::Array<int>& indices,
                        ST* dataT, bool overlapped) const;

    // Rename exodus output file when the problem is resized
    void reNameExodusOutput(const std::string& str);

    /* DAI: old Epetra functions still used by parts of Albany/Trilinos
       Remove when we get to full Tpetra */
#if defined(ALBANY_EPETRA)
    virtual Teuchos::RCP<const Epetra_Map> getMap() const { return map; }
    virtual Teuchos::RCP<const Epetra_Map> getOverlapMap() const { return overlap_map; }
#endif
#if defined(ALBANY_EPETRA)
    virtual Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const;
#endif
#if defined(ALBANY_EPETRA)
    virtual Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const { return graph; }
    virtual Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const { return overlap_graph; }
#endif
#if defined(ALBANY_EPETRA)
    virtual Teuchos::RCP<const Epetra_Map> getNodeMap() const {
      fprintf(stderr,"APF Discretization unsupported call getNodeMap\n");
      abort();
      return Teuchos::RCP<const Epetra_Map>();
    }
#endif
#if defined(ALBANY_EPETRA)
    virtual Teuchos::RCP<Epetra_Vector> getSolutionField(bool overlapped=false) const;
#endif
#if 0 //defined(ALBANY_EPETRA)
    virtual void setResidualField(const Epetra_Vector& residual);
#endif
#if defined(ALBANY_EPETRA)
    virtual void writeSolution(const Epetra_Vector&, const double, const bool);
#endif
#if 0 //defined(ALBANY_EPETRA
    void setSolutionField(const Epetra_Vector&) {
      fprintf(stderr,"APF Discretization unsupported call setSolutionField\n");
      abort();
    }
    void debugMeshWriteNative(const Epetra_Vector&, const char*) {
      fprintf(stderr,"APF Discretization unsupported call debugMeshWriteNative\n");
      abort();
    }
    void debugMeshWrite(const Epetra_Vector&, const char*) {
      fprintf(stderr,"APF Discretization unsupported call debugMeshWrite\n");
      abort();
    }
    // Copy field data from Epetra_Vector to APF
    void setField(
        const char* name,
        const Epetra_Vector& data,
        bool overlapped,
        int offset = 0);

    // Copy field data from APF to Epetra_Vector
    void getField(
        const char* name,
        Epetra_Vector& data,
        bool overlapped,
        int offset = 0) const;

    //! Get field DOF map
    Teuchos::RCP<const Epetra_Map> getMap(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getMap(field_name) not implemented yet");
    }

    //! Get field node map
    Teuchos::RCP<const Epetra_Map> getNodeMap(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getNodeMap(field_name) not implemented yet");
    }

    //! Get field overlapped DOF map
    Teuchos::RCP<const Epetra_Map> getOverlapMap(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getOverlapMap(field_name) not implemented yet");
    }

    //! Get field overlapped node map
    Teuchos::RCP<const Epetra_Map> getOverlapNodeMap(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getOverlapNodeMap(field_name) not implemented yet");
    }

    //! Get field vector from mesh database
    virtual void getField(Epetra_Vector &field_vector, const std::string& field_name) const  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getField(field_vector, field_name) not implemented yet");
    }
    //! Set the field vector into mesh database
    virtual void setField(const Epetra_Vector &field_vector, const std::string& field_name, bool overlapped)  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: setField(field_vector, field_name, overlapped) not implemented yet");
    }
#endif

    //! Get field DOF map
    Teuchos::RCP<const Tpetra_Map> getMapT(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getMapT(field_name) not implemented yet");
    }

    //! Get field node map
    Teuchos::RCP<const Tpetra_Map> getNodeMapT(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getNodeMapT(field_name) not implemented yet");
    }

    //! Get field overlapped DOF map
    Teuchos::RCP<const Tpetra_Map> getOverlapMapT(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getOverlapMapT(field_name) not implemented yet");
    }

    //! Get field overlapped node map
    Teuchos::RCP<const Tpetra_Map> getOverlapNodeMapT(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getOverlapNodeMapT(field_name) not implemented yet");
    }

    //! Get field vector from mesh database
    virtual void getFieldT(Tpetra_Vector &field_vector, const std::string& field_name) const  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getFieldT(field_vector, field_name) not implemented yet");
    }
    //! Set the field vector into mesh database
    virtual void setFieldT(const Tpetra_Vector &field_vector, const std::string& field_name, bool overlapped)  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: setFieldT(field_vector, field_name, overlapped) not implemented yet");
    }

    //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for both scalar and vector fields
    const std::vector<Albany::IDArray>& getElNodeEqID(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getElNodeElID(field_name) not implemented yet");
    }
    //! Get Dof Manager of field field_name
    const Albany::NodalDOFManager& getDOFManager(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getDOFManager(field_name) not implemented yet");
    }

    //! Get Overlapped Dof Manager of field field_name
    const Albany::NodalDOFManager& getOverlapDOFManager(const std::string& field_name) const {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getOverlapDOFManager(field_name) not implemented yet");
    }

    //! Get nodal parameters state info struct
    virtual const Albany::StateInfoStruct& getNodalParameterSIS() const  {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getNodalParameterSIS() not implemented yet");
    }

    //! Get Numbering for layered mesh (mesh structred in one direction)
    Teuchos::RCP<LayeredMeshNumbering<LO> > getLayeredMeshNumbering() {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::APFDiscretization: getLayeredMeshNumbering() not implemented");
      return Teuchos::null;
    }

    //! There can be situations where we want to create a new apf::Mesh2 from
    //! scratch. Clean up everything that depends on the current mesh first,
    //! thereby releasing the mesh.
    virtual void releaseMesh();

    void initTemperatureHack();

    //! Set any FELIX Data
    virtual void setFELIXData() {}

    //! Some evaluators may want access to the underlying apf mesh elements.
    std::vector<std::vector<apf::MeshEntity*> >& getBuckets() {return buckets;}

  private:

    //! Private to prohibit copying
    APFDiscretization(const APFDiscretization&);

    //! Private to prohibit copying
    APFDiscretization& operator=(const APFDiscretization&);

    int nonzeroesPerRow(const int neq) const;
    double monotonicTimeLabel(const double time);

  protected:

    //! Transfer PUMIQPData to APF
    void copyQPScalarToAPF(unsigned nqp, std::string const& state, apf::Field* f);
    void copyQPVectorToAPF(unsigned nqp, std::string const& state, apf::Field* f);
    void copyQPTensorToAPF(unsigned nqp, std::string const& state, apf::Field* f);
    void copyQPStatesToAPF(apf::Field* f, apf::FieldShape* fs, bool copyAll = true);
    void removeQPStatesFromAPF();

    //! Transfer QP Fields from APF to PUMIQPData
    void copyQPScalarFromAPF(unsigned nqp, std::string const& stateName, apf::Field* f);
    void copyQPVectorFromAPF(unsigned nqp, std::string const& stateName, apf::Field* f);
    void copyQPTensorFromAPF(unsigned nqp, std::string const& stateName, apf::Field* f);
    void copyQPStatesFromAPF();

    //! Write stabilized stress out to file
    void saveStabilizedStress();

    // Transfer nodal data to/from APF.
    void copyNodalDataToAPF(const bool copy_all);
    void removeNodalDataFromAPF();

    // ! Split Solution fields
    SolutionLayout solLayout; // solLayout[time_deriv_vector][Field]
    Teuchos::Array<std::string> resNames; // resNames[Field]

  private:

    //! Call stk_io for creating exodus output file
    Teuchos::RCP<Teuchos::FancyOStream> out;

    double previous_time_label;

    // Transformation types for FELIX problems
    enum TRANSFORMTYPE {NONE, ISMIP_HOM_TEST_A};
    TRANSFORMTYPE transform_type;

  protected:

    //! Process APF mesh for Owned nodal quantitites
    void computeOwnedNodesAndUnknowns();
    //! Process APF mesh for Overlap nodal quantitites
    void computeOverlapNodesAndUnknowns();
    //! Process APF mesh for CRS Graphs
    void computeGraphs();
    //! Process APF mesh for Workset/Bucket Info
    void computeWorksetInfo();
    //! Process APF mesh for NodeSets
    void computeNodeSets();
    //! Process APF mesh for SideSets
    void computeSideSets();
    //! Re-initialize Time after adaptation
    void initTimeFromParamLib(Teuchos::RCP<ParamLib> paramLib);

    //! Output object
    PUMIOutput* meshOutput;

    //! Stk Mesh Objects

#if defined(ALBANY_EPETRA)
    //! Epetra communicator
    Teuchos::RCP<const Epetra_Comm> comm;
#endif

   //! Tpetra communicator and Kokkos node
    Teuchos::RCP<const Teuchos::Comm<int> > commT;

    //! Node map
    Teuchos::RCP<const Tpetra_Map> node_mapT;

    //! Unknown Map
    Teuchos::RCP<const Tpetra_Map> mapT;
#if defined(ALBANY_EPETRA)
    Teuchos::RCP<Epetra_Map> map;
#endif

    //! Overlapped unknown map, and node map
    Teuchos::RCP<const Tpetra_Map> overlap_mapT;
#if defined(ALBANY_EPETRA)
    Teuchos::RCP<Epetra_Map> overlap_map;
#endif
    Teuchos::RCP<const Tpetra_Map> overlap_node_mapT;

    //! Jacobian matrix graph
    Teuchos::RCP<Tpetra_CrsGraph> graphT;
#if defined(ALBANY_EPETRA)
    Teuchos::RCP<Epetra_CrsGraph> graph;
#endif

    //! Overlapped Jacobian matrix graph
    Teuchos::RCP<Tpetra_CrsGraph> overlap_graphT;
#if defined(ALBANY_EPETRA)
    Teuchos::RCP<Epetra_CrsGraph> overlap_graph;
#endif

    //! Number of equations (and unknowns) per node
    const unsigned int neq;

    //! node sets stored as std::map(string ID, int vector of GIDs)
    Albany::NodeSetList nodeSets;
    Albany::NodeSetGIDsList nodeSetGIDs; // not used
    Albany::NodeSetCoordList nodeSetCoords;

    //! side sets stored as std::map(string ID, SideArray classes) per workset (std::vector across worksets)
    std::vector<Albany::SideSetList> sideSets;

    // Side set discretizations related structures (not supported but needed for getters return values)
    std::map<std::string,Teuchos::RCP<Albany::AbstractDiscretization> > sideSetDiscretizations;
    std::map<std::string,std::map<GO,GO> >                              sideToSideSetCellMap;
    std::map<std::string,std::map<GO,std::vector<int> > >               sideNodeNumerationMap;

    //! Connectivity array [workset, element, local-node, Eq] => LID
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type wsElNodeEqID;

    //! Connectivity array [workset, element, local-node] => GID
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type wsElNodeID;

    mutable Teuchos::ArrayRCP<double> coordinates;
    Teuchos::RCP<Tpetra_MultiVector> coordMV;
    Albany::WorksetArray<std::string>::type wsEBNames;
    Albany::WorksetArray<int>::type wsPhysIndex;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type coords;
    Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type sphereVolume;
    Albany::WorksetArray<Teuchos::ArrayRCP<double*> >::type latticeOrientation;

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

    Teuchos::RCP<Albany::APFMeshStruct> meshStruct;

    bool interleavedOrdering;

    std::vector< std::vector<apf::MeshEntity*> > buckets; // bucket of elements

    // storage to save the node coordinates of the nodesets visible to this PE
    std::map<std::string, std::vector<double> > nodeset_node_coords;

    // Needed to pass coordinates to ML.
    Teuchos::RCP<Albany::RigidBodyModes> rigidBodyModes;

    // counter for limiting data writes to output file
    int outputInterval;

    // counter for the continuation step number
    int continuationStep;

    // Mesh adaptation stuff.
    Teuchos::RCP<AAdapt::rc::Manager> rcm;

  };

}

#endif // ALBANY_PUMIDISCRETIZATION_HPP

