//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_APF_DISCRETIZATION_HPP
#define ALBANY_APF_DISCRETIZATION_HPP

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DiscretizationUtils.hpp"

#include "Albany_APFMeshStruct.hpp"
#include "Albany_PUMIOutput.hpp"

#include "Albany_NullSpaceUtils.hpp"
#include "Albany_SacadoTypes.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_CompilerCodeTweakMacros.hpp"
#include "Albany_Utils.hpp" 

#include <vector>
#include <functional>
#include <stdexcept>

namespace Albany {

class APFDiscretization : public AbstractDiscretization {
public:

  //! Constructor
  APFDiscretization(const Teuchos::RCP<APFMeshStruct> meshStruct_in,
                    const Teuchos::RCP<const Teuchos_Comm>& comm,
                    const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null);

  APFDiscretization(const APFDiscretization&) = delete;
  APFDiscretization& operator=(const APFDiscretization&) = delete;

  //! Destructor
  virtual ~APFDiscretization();

  //! Initialize this class
  void init();

  //! Set any restart data
  virtual void setRestartData() {}

  //! Get node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace> getNodeVectorSpace        () const override { return m_node_vs; }
  Teuchos::RCP<const Thyra_VectorSpace> getOverlapNodeVectorSpace () const override { return m_overlap_node_vs; }

  //! Get solution DOF vector space (owned and overlapped).
  Teuchos::RCP<const Thyra_VectorSpace> getVectorSpace        () const override { return m_vs; }
  Teuchos::RCP<const Thyra_VectorSpace> getOverlapVectorSpace () const override { return m_overlap_vs; }

  //! Get Field node vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace> getNodeVectorSpace (const std::string& /* field_name */) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Albany::APFDiscretization: getNodeVectorSpace(field_name) not implemented yet");
    TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
  }

  Teuchos::RCP<const Thyra_VectorSpace> getOverlapNodeVectorSpace (const std::string& /* field_name */) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Albany::APFDiscretization: getOverlapNodeVectorSpace(field_name) not implemented yet");
    TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
  }

  //! Get Field vector space (owned and overlapped)
  Teuchos::RCP<const Thyra_VectorSpace> getVectorSpace (const std::string& /* field_name */) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Albany::APFDiscretization: getVectorSpace(field_name) not implemented yet");
    TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
  }

  Teuchos::RCP<const Thyra_VectorSpace> getOverlapVectorSpace (const std::string& /* field_name */) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Albany::APFDiscretization: getOverlapVectorSpace(field_name) not implemented yet");
    TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
  }

  //! Create a Jacobian operator (owned and overlapped)
  Teuchos::RCP<Thyra_LinearOp> createJacobianOp        () const override { return m_jac_factory->createOp();         }
  Teuchos::RCP<Thyra_LinearOp> createOverlapJacobianOp () const override { return m_overlap_jac_factory->createOp(); }

#ifdef ALBANY_AERAS
  //! Create implicit Jacobian operator (owned and overlapped) (for Aeras)
  Teuchos::RCP<Thyra_LinearOp> createImplicitJacobianOp        () const override { return m_jac_factory->createOp();         }
  Teuchos::RCP<Thyra_LinearOp> createImplicitOverlapJacobianOp () const override { return m_overlap_jac_factory->createOp(); }
#endif

  bool isExplicitScheme() const override { return false; }

  //! Process coords for ML
  void setupMLCoords();

  //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
  const NodeSetList& getNodeSets() const override { return nodeSets; }
  const NodeSetCoordList& getNodeSetCoords() const override { return nodeSetCoords; }
  // not used; just completing concrete impl
  const NodeSetGIDsList& getNodeSetGIDs() const override { return nodeSetGIDs; }

  //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
  const SideSetList& getSideSets(const int workset) const override { return sideSets[workset]; }

  //! Get connectivity map from elementGID to workset
  WsLIDList& getElemGIDws() override { return elemGIDws; }
  const WsLIDList& getElemGIDws() const override { return elemGIDws; }

  //! Get map from (Ws, El, Local Node, Eqn) -> dof LID
  const Conn& getWsElNodeEqID() const override { return wsElNodeEqID; }

  //! Get map from (Ws, El, Local Node) -> NodeGID
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type& getWsElNodeID() const override {
    return wsElNodeID;
  }

  //! Get coordinate vector (overlap map, interleaved)
  const Teuchos::ArrayRCP<double>& getCoordinates() const override;
  //! Set coordinate vector (overlap map, interleaved)
  void setCoordinates(const Teuchos::ArrayRCP<const double>& c) override;
  void setReferenceConfigurationManager(const Teuchos::RCP<AAdapt::rc::Manager>& rcm) override;

#ifdef ALBANY_CONTACT
//! Get the contact manager
  Teuchos::RCP<const ContactManager> getContactManager() const override { return contactManager; }
#endif

  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const override {
    return coords;
  }

  const WorksetArray<Teuchos::ArrayRCP<double> >::type& getSphereVolume() const override {
    return sphereVolume;
  }

  const WorksetArray<Teuchos::ArrayRCP<double*> >::type& getLatticeOrientation() const override {
    return latticeOrientation;
  }

  //! Print coords for debugging
  void printCoords() const override;

  //! Get sideSet discretizations map
  const SideSetDiscretizationsType& getSideSetDiscretizations () const override
  {
    //Warning: returning empty SideSetDiscretization for now.
    return sideSetDiscretizations;
  }

  //! Get the map side_id->side_set_elem_id
  const std::map<std::string,std::map<GO,GO> >& getSideToSideSetCellMap () const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Functionality not supported by APF discretization.\n");
    return sideToSideSetCellMap;
  }

  //! Get the map side_node_id->side_set_cell_node_id
  const std::map<std::string,std::map<GO,std::vector<int>>>& getSideNodeNumerationMap () const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Functionality not supported by APF discretization.\n");
    return sideNodeNumerationMap;
  }

  //! Get number of spatial dimensions
  int getNumDim() const override { return meshStruct->numDim; }

  virtual Teuchos::RCP<const Teuchos_Comm> getComm() const { return comm; }

  //! Get number of total DOFs per node
  int getNumEq() const override { return neq; }

  //! Set stateArrays
  void setStateArrays(StateArrays& sa) override {
    stateArrays = sa;
    return;
  }

  StateArrays& getStateArrays() override {return stateArrays;}

  //! Retrieve Vector (length num worksets) of element block names
  const WorksetArray<std::string>::type& getWsEBNames() const override { return wsEBNames; }
  //! Retrieve Vector (length num worksets) of physics set index
  const WorksetArray<int>::type&  getWsPhysIndex() const override { return wsPhysIndex; }

  virtual void writeMeshDebug (const std::string& filename);

  // Retrieve mesh struct
  Teuchos::RCP<APFMeshStruct> getAPFMeshStruct() {return meshStruct;}
  Teuchos::RCP<AbstractMeshStruct> getMeshStruct() const override {return meshStruct;}

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const override {return meshStruct->hasRestartSolution;}

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const override {return meshStruct->restartDataTime;}

  // Before mesh modification, qp data may be needed for solution transfer
  void attachQPData();

  // After mesh modification, qp data needs to be removed
  void detachQPData();

  // Communicates most APF data structures to Albany,
  // including worksets, sidesets, nodesets, blocks, graphs, etc.
  // This function is called by the constructor and by updateMesh.
  void initMesh();

  // After mesh modification, calls initMesh() plus two other things.
  // First, integration/quadrature point data must be copied into state arrays.
  // Second, the parameter library is used to set Time on each workset.
  void updateMesh(bool shouldTransferIPData, Teuchos::RCP<ParamLib> paramLib);

  // Function that transforms a mesh of a unit cube (for LandIce problems)
  // not supported in PUMI now
  void transformMesh(){}

  // this is called with both LO's and GO's to compute a dof number
  // based on a node number and an equation number
  GO getDOF(const GO inode, const int entry, int total_comps = -1) const
  {
    if (interleavedOrdering) {
      if (total_comps == -1) {
        total_comps = neq;
      }
      return inode * total_comps + entry;
    } else {
      return inode + numOwnedNodes*entry;
    }
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

  //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for both scalar and vector fields
  const std::vector<IDArray>& getElNodeEqID(const std::string& /* field_name */) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "APFDiscretization: getElNodeElID(field_name) not implemented yet");
  }
  //! Get Dof Manager of field field_name
  const NodalDOFManager& getDOFManager(const std::string& /* field_name */) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "APFDiscretization: getDOFManager(field_name) not implemented yet");
  }

  //! Get Overlapped Dof Manager of field field_name
  const NodalDOFManager& getOverlapDOFManager(const std::string& /* field_name */) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "APFDiscretization: getOverlapDOFManager(field_name) not implemented yet");
  }

  //! Get nodal parameters state info struct
  virtual const StateInfoStruct& getNodalParameterSIS() const  override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "APFDiscretization: getNodalParameterSIS() not implemented yet");
  }

  //! Get Numbering for layered mesh (mesh structured in one direction)
  Teuchos::RCP<LayeredMeshNumbering<LO>> getLayeredMeshNumbering() const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "APFDiscretization: getLayeredMeshNumbering() not implemented");
    return Teuchos::null;
  }

  void initTemperatureHack();

  //! Set any LandIce Data
  virtual void setLandIceData() {}

  //! Some evaluators may want access to the underlying apf mesh elements.
  std::vector<std::vector<apf::MeshEntity*> >& getBuckets() {return buckets;}

  //! Get the solution vector layouts
  SolutionLayout const& getSolutionLayout() { return solLayout; }

  //! Get the residual field names
  Teuchos::Array<std::string> const& getResNames() { return resNames; }

  //! Get the APF owned nodes
  apf::DynamicArray<apf::Node> const& getOwnedNodes() { return ownedNodes; }



  /* DAI: old Epetra functions still used by parts of Albany/Trilinos
     Remove when we get to full Tpetra */
// #if defined(ALBANY_EPETRA)
//   virtual Teuchos::RCP<const Epetra_Map> getMap() const override { return map; }
//   virtual Teuchos::RCP<const Epetra_Map> getOverlapMap() const override { return overlap_map; }
//   virtual Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const override;
//   virtual Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const override { return graph; }
//   virtual Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const override { return overlap_graph; }
//   virtual Teuchos::RCP<const Epetra_Map> getNodeMap() const override {
//     fprintf(stderr,"APF Discretization unsupported call getNodeMap\n");
//     abort();
//     return Teuchos::RCP<const Epetra_Map>();
//   }
// #endif

  // --- Get/set solution/residual/field vectors to/from mesh --- //

  Teuchos::RCP<Thyra_Vector>      getSolutionField (const bool overlapped) const override;
  Teuchos::RCP<Thyra_MultiVector> getSolutionMV    (const bool overlapped) const override;

#if defined(ALBANY_LCM)
  void setResidualField (const Thyra_Vector& residual) override;
#endif

  void getField (Thyra_Vector& /* field_vector */, const std::string& /* field_name */) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Method not yet implemented in APFDiscretization.\n";)
  }
  void setField (const Thyra_Vector& /* field_vector */, const std::string& /* field_name */, const bool /* overlapped */) override {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Method not yet implemented in APFDiscretization.\n";)
  }

  // --- Methods to write solution in the output file --- //

  void writeSolution (const Thyra_Vector& solution,
                      const double time, const bool overlapped) override;
  void writeSolution (const Thyra_Vector& solution,
                      const Thyra_Vector& solution_dot,
                      const double time, const bool overlapped) override;
  void writeSolution (const Thyra_Vector& solution,
                      const Thyra_Vector& solution_dot,
                      const Thyra_Vector& solution_dotdot,
                      const double time, const bool overlapped) override;
  void writeSolutionMV (const Thyra_MultiVector& solution,
                        const double time, const bool overlapped) override;

  //! Write the solution to the mesh database.
  void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                    const double time, const bool overlapped) override;
  void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                    const Thyra_Vector& solution_dot,
                                    const double time, const bool overlapped) override;
  void writeSolutionToMeshDatabase (const Thyra_Vector& solution,
                                    const Thyra_Vector& solution_dot,
                                    const Thyra_Vector& solution_dotdot,
                                    const double time, const bool overlapped) override;
  void writeSolutionMVToMeshDatabase (const Thyra_MultiVector &solution,
                                      const double time, const bool overlapped) override;

  //! Write the solution to file. Must call writeSolution first.
  void writeSolutionToFile (const Thyra_Vector &solution,
                            const double time, const bool overlapped) override;
  void writeSolutionMVToFile (const Thyra_MultiVector &solution,
                              const double time, const bool overlapped) override;


  void writeAnySolutionToMeshDatabase(const ST* soln, const int index, const bool overlapped);
  void writeAnySolutionToFile(const double time);

  void writeRestartFile(const double time);
    
  WorksetArray<Teuchos::ArrayRCP<double*>>::type const& getBoundaryIndicator() const 
  {
    ALBANY_ASSERT(boundary_indicator.is_null() == false);
    return boundary_indicator;
  };  

  void printElemGIDws(std::ostream& os) const 
  {//do nothing 
  }; 

  std::map<std::pair<int, int>, GO>
  getElemWsLIDGIDMap() const 
  {
	throw std::runtime_error("Calling getElemWsLIDGIDMap() in Albany_APFDiscretization.hpp");
  };

private:

  int nonzeroesPerRow(const int neq) const;
  double monotonicTimeLabel(const double time);

public:

  //! Transfer PUMIQPData to APF
  void copyQPScalarToAPF(unsigned nqp, std::string const& state, apf::Field* f);
  void copyQPVectorToAPF(unsigned nqp, std::string const& state, apf::Field* f);
  void copyQPTensorToAPF(unsigned nqp, std::string const& state, apf::Field* f);
  void copyQPStatesToAPF(apf::FieldShape* fs, bool copyAll = true);
  void removeQPStatesFromAPF();

  //! Transfer QP Fields from APF to PUMIQPData
  void copyQPScalarFromAPF(unsigned nqp, std::string const& stateName, apf::Field* f);
  void copyQPVectorFromAPF(unsigned nqp, std::string const& stateName, apf::Field* f);
  void copyQPTensorFromAPF(unsigned nqp, std::string const& stateName, apf::Field* f);
  void copyQPStatesFromAPF();

protected:

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

  // Transformation types for LandIce problems
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
  void forEachNodeSetNode(std::function<void(size_t, apf::StkModel*)> fn);
  //! Process APF mesh for SideSets
  void computeSideSets();
  //! Re-initialize Time after adaptation
  void initTimeFromParamLib(Teuchos::RCP<ParamLib> paramLib);

  //! Output object
  PUMIOutput* meshOutput;

  //! Teuchos communicator
  Teuchos::RCP<const Teuchos_Comm>        comm;

  //! Unknown map and node map
  Teuchos::RCP<const Thyra_VectorSpace>   m_vs;
  Teuchos::RCP<const Thyra_VectorSpace>   m_node_vs;

  //! Overlapped unknown map and node map
  Teuchos::RCP<const Thyra_VectorSpace>   m_overlap_vs;
  Teuchos::RCP<const Thyra_VectorSpace>   m_overlap_node_vs;

  //! Jacobian matrix graph proxy (owned and overlap)
  Teuchos::RCP<ThyraCrsMatrixFactory> m_jac_factory;
  Teuchos::RCP<ThyraCrsMatrixFactory> m_overlap_jac_factory;

  //! Number of equations (and unknowns) per node
  const unsigned int neq;

  //! node sets stored as std::map(string ID, int vector of GIDs)
  NodeSetList nodeSets;
  NodeSetGIDsList nodeSetGIDs; // not used
  NodeSetCoordList nodeSetCoords;

  //! side sets stored as std::map(string ID, SideArray classes) per workset (std::vector across worksets)
  std::vector<SideSetList> sideSets;

  // Side set discretizations related structures (not supported but needed for getters return values)
  std::map<std::string,Teuchos::RCP<AbstractDiscretization> >   sideSetDiscretizations;
  std::map<std::string,std::map<GO,GO> >                        sideToSideSetCellMap;
  std::map<std::string,std::map<GO,std::vector<int> > >         sideNodeNumerationMap;

  //! Connectivity array [workset, element, local-node, Eq] => LID
  Conn wsElNodeEqID;

  //! Connectivity array [workset, element, local-node] => GID
  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type wsElNodeID;

  mutable Teuchos::ArrayRCP<double>     coordinates;
  Teuchos::RCP<Thyra_MultiVector>       coordMV;
  WorksetArray<std::string>::type       wsEBNames;
  WorksetArray<int>::type               wsPhysIndex;
  WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type coords;
  WorksetArray<Teuchos::ArrayRCP<double> >::type sphereVolume;
  WorksetArray<Teuchos::ArrayRCP<double*> >::type latticeOrientation;

#ifdef ALBANY_CONTACT
  Teuchos::RCP<const ContactManager> contactManager;
#endif

  //! Connectivity map from elementGID to workset and LID in workset
  WsLIDList  elemGIDws;

  // States: vector of length num worksets of a map from field name to shards array
  StateArrays stateArrays;

  apf::GlobalNumbering* globalNumbering;
  apf::GlobalNumbering* elementNumbering;

  //! list of all overlap nodes, saved for setting solution
  apf::DynamicArray<apf::Node> overlapNodes;
  apf::DynamicArray<apf::Node> ownedNodes;

  //! Number of elements on this processor
  int numOwnedNodes;
  int numOverlapNodes;

  Teuchos::RCP<APFMeshStruct> meshStruct;

  bool interleavedOrdering;

  std::vector< std::vector<apf::MeshEntity*> > buckets; // bucket of elements

  // storage to save the node coordinates of the nodesets visible to this PE
  std::map<std::string, std::vector<double> > nodeset_node_coords;

  // Needed to pass coordinates to ML.
  Teuchos::RCP<RigidBodyModes> rigidBodyModes;

  // counter for limiting data writes to output file
  int outputInterval;

  // counter for the continuation step number
  int continuationStep;

  // Mesh adaptation stuff.
  Teuchos::RCP<AAdapt::rc::Manager> rcm;
    
  WorksetArray<Teuchos::ArrayRCP<double*>>::type boundary_indicator;
};

} // namespace Albany

#endif // ALBANY_APF_DISCRETIZATION_HPP
