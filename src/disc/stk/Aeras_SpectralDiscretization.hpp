//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AERAS_SPECTRALDISCRETIZATION_HPP
#define AERAS_SPECTRALDISCRETIZATION_HPP

#include <vector>
#include <utility>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"


#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_DataTypes.hpp"

#if defined(ALBANY_EPETRA)
#include "Epetra_Comm.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"
#endif

#include "Albany_NullSpaceUtils.hpp"

// Start of STK stuff
#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>
#ifdef ALBANY_SEACAS
  #include <stk_io/StkMeshIoBroker.hpp>
#endif

#include "Shards_CellTopology.hpp"
#include "Aeras_SpectralOutputSTKMeshStruct.hpp"

// Uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace Aeras
{

  struct AerasMeshSpectStruct
  {
    Teuchos::RCP<Albany::MeshSpecsStruct>
    createAerasMeshSpecs(
      const Teuchos::RCP<Albany::MeshSpecsStruct>& orig_mesh_specs_struct, 
      const int points_per_edge) 
    {
#ifdef OUTPUT_TO_SCREEN
      std::cout << "DEBUG: in AerasMeshSpectStruct!  Element Degree =  "
                << points_per_edge << std::endl;
#endif 
      //get data from original STK Mesh struct
      CellTopologyData orig_ctd = orig_mesh_specs_struct->ctd; 
      std::string orig_name = orig_ctd.name;
      size_t len      = orig_name.find("_");
      if (len != std::string::npos) orig_name = orig_name.substr(0,len);
      TEUCHOS_TEST_FOR_EXCEPTION(
        (orig_name != "ShellQuadrilateral") && (orig_name!= "Quadrilateral")
        && (orig_name != "Line"), 
        Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error!  Attempting to enrich a non-quadrilateral element "
        << "(" << orig_name << ")!  Aeras::SpectralDiscretization is currently "
        << "implemented only for " << "Quadrilateral, ShellQuadrilateral and "
        << "Line elements." << std::endl); 
#ifdef OUTPUT_TO_SCREEN
      std::cout << "DEBUG: original ctd name = " << orig_name << std::endl; 
#endif 
      int orig_numDim = orig_mesh_specs_struct->numDim;
      int orig_cubatureDegree = orig_mesh_specs_struct->cubatureDegree;
      // Node Sets Names
      std::vector<std::string> orig_nsNames = orig_mesh_specs_struct->nsNames;
      // Side Sets Names
      std::vector<std::string> orig_ssNames = orig_mesh_specs_struct->ssNames;
      int orig_worksetSize = orig_mesh_specs_struct->worksetSize;
      //Element block name for the EB that this struct corresponds to
      std::string orig_ebName = orig_mesh_specs_struct->ebName;
      std::map<std::string, int>& orig_ebNameToIndex = 
        orig_mesh_specs_struct->ebNameToIndex;
      bool orig_interleavedOrdering =
        orig_mesh_specs_struct->interleavedOrdering;
      bool orig_sepEvalsByEB = orig_mesh_specs_struct->sepEvalsByEB;
      const Intrepid::EIntrepidPLPoly orig_cubatureRule =
        orig_mesh_specs_struct->cubatureRule;
      // Create enriched MeshSpecsStruct object, to be returned.  It
      // will have the same everything as the original mesh struct
      // except a CellTopologyData (ctd) with a different name and
      // node_count (and dimension?).  New (enriched) CellTopologyData
      // is same as original (unenriched) cell topology data (ctd),
      // but with a different node_count, vertex_count and name.
      CellTopologyData new_ctd = orig_ctd; 
      //overwrite node_count, vertex_count and name of the original ctd.
      int np; 
      if (orig_name == "ShellQuadrilateral" || orig_name == "Quadrilateral") 
        np = points_per_edge*points_per_edge; 
      else if (orig_name == "Line") 
        np = points_per_edge;
      new_ctd.node_count = np;
      // Assumes vertex_count = node_count for ctd, which is the case
      // for isoparametric finite elements.
      new_ctd.vertex_count = np;

      // Used to convert int to string  
      std::ostringstream convert;
      convert << np; 
      std::string new_name = "Spectral" + orig_name + '_' + convert.str();
      // The following seems to be necessary b/c setting new_ctd.name
      // = new_name.c_str() does not work.
      char* new_name_char = new char[new_name.size() + 1]; 
      std::copy(new_name.begin(), new_name.end(), new_name_char);
      new_name_char[new_name.size()] = '\0';
      new_ctd.name = new_name_char;   
#ifdef OUTPUT_TO_SCREEN
      std::cout << "DEBUG: new_ctd.name = " << new_ctd.name << std::endl; 
#endif
      // Create and return Albany::MeshSpecsStruct object based on the
      // new (enriched) ctd.
      return Teuchos::rcp(new Albany::MeshSpecsStruct(new_ctd,
                                                      orig_numDim,
                                                      orig_cubatureDegree,
                                                      orig_nsNames,
                                                      orig_ssNames,
                                                      orig_worksetSize,
                                                      orig_ebName,
                                                      orig_ebNameToIndex,
                                                      orig_interleavedOrdering,
                                                      orig_sepEvalsByEB,
                                                      orig_cubatureRule));
      delete [] new_name_char;
    }
  };

#if defined(ALBANY_EPETRA)
  typedef shards::Array<GO, shards::NaturalOrder> GIDArray;

  struct DOFsStruct
  {
    Teuchos::RCP<Epetra_Map> node_map;
    Teuchos::RCP<Epetra_Map> overlap_node_map;
    Teuchos::RCP<Epetra_Map> map;
    Teuchos::RCP<Epetra_Map> overlap_map;
    Albany::NodalDOFManager dofManager;
    Albany::NodalDOFManager overlap_dofManager;
    std::vector< std::vector<LO> > wsElNodeEqID_rawVec;
    std::vector<Albany::IDArray> wsElNodeEqID;
    std::vector< std::vector<GO> > wsElNodeID_rawVec;
    std::vector<GIDArray> wsElNodeID;
  };

  struct NodalDOFsStructContainer
  {
    typedef std::map<std::pair<std::string,int>, DOFsStruct >  MapOfDOFsStructs;

    MapOfDOFsStructs mapOfDOFsStructs;
    std::map<std::string, MapOfDOFsStructs::const_iterator> fieldToMap;
    const DOFsStruct& getDOFsStruct(const std::string& field_name) const
    {
      // TODO: handole errors
      return fieldToMap.find(field_name)->second->second;
    }

    void addEmptyDOFsStruct(const std::string& field_name,
                            const std::string& meshPart,
                            int numComps)
    {
      if(numComps != 1)
        mapOfDOFsStructs.insert(make_pair(make_pair(meshPart,1),DOFsStruct()));
      fieldToMap[field_name] =
        mapOfDOFsStructs.insert(make_pair(make_pair(meshPart,numComps),
                                          DOFsStruct())).first;
    }
  };
#endif // ALBANY_EPETRA

  class SpectralDiscretization : public Albany::AbstractDiscretization
  {
  public:

    //! Constructor
    SpectralDiscretization(
       const Teuchos::RCP<Teuchos::ParameterList>& discParams,
       Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct,
       const Teuchos::RCP<const Teuchos_Comm>& commT,
       const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes=Teuchos::null);

    //! Destructor
    ~SpectralDiscretization();

#if defined(ALBANY_EPETRA)
    //! Get Epetra DOF map
    Teuchos::RCP<const Epetra_Map> getMap() const;
    
    //! Get overlapped node map
    Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const;

    //! Get field overlapped node map
    Teuchos::RCP<const Epetra_Map>
    getOverlapNodeMap(const std::string& field_name) const;
#endif
    //! Get Tpetra DOF map
    Teuchos::RCP<const Tpetra_Map> getMapT() const;

    //! Get Tpetra overlapped DOF map
    Teuchos::RCP<const Tpetra_Map> getOverlapMapT() const;

#if defined(ALBANY_EPETRA)
    //! Get field DOF map
    Teuchos::RCP<const Epetra_Map> getMap(const std::string& field_name) const;

#endif

#if defined(ALBANY_EPETRA)
    //! Get Epetra Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const;
#endif
    //! Get Tpetra Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getJacobianGraphT() const;

#if defined(ALBANY_EPETRA)
    //! Get Epetra overlap Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const;
#endif
    //! Get Tpetra overlap Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getOverlapJacobianGraphT() const;

#if defined(ALBANY_EPETRA)
    //! Get field node map
    Teuchos::RCP<const Epetra_Map> getNodeMap() const;
    //! Get field node map
    Teuchos::RCP<const Epetra_Map> getNodeMap(const std::string& field_name) const;
    //! Get field overlapped DOF map
    Teuchos::RCP<const Epetra_Map> getOverlapMap() const;
    //! Get field overlapped DOF map
    Teuchos::RCP<const Epetra_Map> getOverlapMap(const std::string& field_name) const;
#endif
    //! Get Tpetra Node map
    Teuchos::RCP<const Tpetra_Map> getNodeMapT() const; 
    Teuchos::RCP<const Tpetra_Map> getOverlapNodeMapT() const;

    //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
    const Albany::NodeSetList& getNodeSets() const
    {
      return nodeSets;
    };

    const Albany::NodeSetCoordList& getNodeSetCoords() const
    {
      return nodeSetCoords;
    };
    const Albany::NodeSetGIDsList& getNodeSetGIDs() const 
    { 
      return nodeSetGIDs; 
    };

    //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
    const Albany::SideSetList& getSideSets(const int workset) const
    {
      return sideSets[workset];
    };

    //! Get connectivity map from elementGID to workset
    Albany::WsLIDList& getElemGIDws()
    {
      return elemGIDws;
    };

    //! Get map from (Ws, El, Local Node) -> NodeLID
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type&
    getWsElNodeEqID() const;

    //! Get map from (Ws, Local Node) -> NodeGID
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    getWsElNodeID() const;

#if defined(ALBANY_EPETRA)
    //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID,
    //! works for both scalar and vector fields
    const std::vector<Albany::IDArray>&
    getElNodeEqID(const std::string& field_name) const
    {
      return nodalDOFsStructContainer.getDOFsStruct(field_name).wsElNodeEqID;
    }

    const Albany::NodalDOFManager&
    getDOFManager(const std::string& field_name) const
    {
      return nodalDOFsStructContainer.getDOFsStruct(field_name).dofManager;
    }
    
    const Albany::NodalDOFManager& 
    getOverlapDOFManager(const std::string& field_name) const
    {
      return nodalDOFsStructContainer.getDOFsStruct(field_name).overlap_dofManager;
    }
#endif



    //! Retrieve coodinate vector (num_used_nodes * 3)
    const Teuchos::ArrayRCP<double>& getCoordinates() const;

    //! Set coordinate vector (num_used_nodes * 3)
    void setCoordinates(const Teuchos::ArrayRCP<const double>& c);

    void
    setReferenceConfigurationManager(const Teuchos::RCP<AAdapt::rc::Manager>& rcm);

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
    getCoords() const;

    const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type&
    getSphereVolume() const;

    //! Print the coordinates for debugging
    void printCoords() const;
    void printConnectivity(bool printEdges=false) const;
    void printCoordsAndGIDs() const; 

    //! Get stateArrays
    Albany::StateArrays& getStateArrays() {return stateArrays;}

    //! Get nodal parameters state info struct
    const Albany::StateInfoStruct& getNodalParameterSIS() const
    {
      return stkMeshStruct->getFieldContainer()->getNodalParameterSIS();
    }

    //! Retrieve Vector (length num worksets) of element block names
    const Albany::WorksetArray<std::string>::type&  getWsEBNames() const;
    //! Retrieve Vector (length num worksets) of physics set index
    const Albany::WorksetArray<int>::type&  getWsPhysIndex() const;

#if defined(ALBANY_EPETRA)
    void writeSolution(const Epetra_Vector& soln,
                       const double time,
                       const bool overlapped = false);
#endif
   
   void writeSolutionT(const Tpetra_Vector& solnT,
                       const double time,
                       const bool overlapped = false);

   void writeSolutionToMeshDatabaseT(const Tpetra_Vector &solutionT,
                                     const double time,
                                     const bool overlapped = false);

   void writeSolutionToFileT(const Tpetra_Vector& solnT,
                             const double time,
                             const bool overlapped = false);

#if defined(ALBANY_EPETRA) 
    Teuchos::RCP<Epetra_Vector>
    getSolutionField(const bool overlapped=false) const;
#endif
    //Tpetra analog
    Teuchos::RCP<Tpetra_Vector>
    getSolutionFieldT(const bool overlapped=false) const;

    int getSolutionFieldHistoryDepth() const;
#if defined(ALBANY_EPETRA)
    Teuchos::RCP<Epetra_MultiVector>
    getSolutionFieldHistory() const;
    Teuchos::RCP<Epetra_MultiVector>
    getSolutionFieldHistory(int maxStepCount) const;
    void getSolutionFieldHistory(Epetra_MultiVector &result) const;

    void setResidualField(const Epetra_Vector& residual);
#endif
    //Tpetra analog
    void setResidualFieldT(const Tpetra_Vector& residualT);

    // Retrieve mesh struct
    Teuchos::RCP<Albany::AbstractSTKMeshStruct> getSTKMeshStruct()
    {
      return stkMeshStruct;
    }

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const
    {
      return stkMeshStruct->hasRestartSolution();
    }

    //! STK supports MOR
    virtual bool supportsMOR() const
    {
      return true;
    }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const
    {
      return stkMeshStruct->restartDataTime();
    }

    //! After mesh modification, need to update the element
    //! connectivity and nodal coordinates
    void updateMesh(bool shouldTransferIPData = false);

    //! Function that transforms an STK mesh of a unit cube (for FELIX problems)
    void transformMesh();

    //! Close current exodus file in stk_io and create a new one for an adapted mesh and new results
    void reNameExodusOutput(std::string& filename);

   //! Get number of spatial dimensions
    int getNumDim() const
    {
      return stkMeshStruct->numDim;
    }

    //! Get number of total DOFs per node
    int getNumEq() const
    {
      return neq;
    }

    //! Locate nodal dofs in non-overlapping vectors using local indexing
    int getOwnedDOF(const int inode,
                    const int eq) const;

    //! Locate nodal dofs in overlapping vectors using local indexing
    int getOverlapDOF(const int inode,
                      const int eq) const;

    //! Locate nodal dofs using global indexing
    GO getGlobalDOF(const GO inode,
                    const int eq) const;


    //! Used when NetCDF output on a latitude-longitude grid is
    //! requested.  Each struct contains a latitude/longitude index
    //! and it's parametric coordinates in an element.
    struct interp
    {
      std::pair<double  , double  > parametric_coords ;
      std::pair<unsigned, unsigned> latitude_longitude;
    };

    const stk::mesh::MetaData& getSTKMetaData()
    {
      return metaData;
    }

    const stk::mesh::BulkData& getSTKBulkData()
    {
      return bulkData;
    }

    Teuchos::RCP<Albany::LayeredMeshNumbering<LO> > getLayeredMeshNumbering(){
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::SpectralDiscretization: getLayeredMeshNumbering() not implemented");
      return Teuchos::null;
    }

  private:

    //! Private to prohibit copying
    SpectralDiscretization(const SpectralDiscretization&);

    //! Private to prohibit copying
    SpectralDiscretization& operator=(const SpectralDiscretization&);

    inline GO gid(const stk::mesh::Entity node) const;

#if defined(ALBANY_EPETRA)
    // Copy values from STK Mesh field to given Epetra_Vector
    void getSolutionField(Epetra_Vector &result,
                          bool overlapped=false) const;
#endif
    // Copy values from STK Mesh field to given Tpetra_Vector
    void getSolutionFieldT(Tpetra_Vector &resultT,
                           bool overlapped=false) const;

#if defined(ALBANY_EPETRA)
    //! Copy field from STK Mesh field to given Epetra_Vector
    void getField(Epetra_Vector &field_vector,
                  const std::string& field_name) const;

    // Copy field vector into STK Mesh field
    void setField(const Epetra_Vector &field_vector,
                  const std::string& field_name,
                  bool overlapped=false);

    Teuchos::RCP<Epetra_MultiVector>
    getSolutionFieldHistoryImpl(int stepCount) const;
    void getSolutionFieldHistoryImpl(Epetra_MultiVector &result) const;

    // Copy solution vector from Epetra_Vector into STK Mesh
    // Here soln is the local (non overlapped) solution
    void setSolutionField(const Epetra_Vector& soln);
#endif
    //Tpetra version of above
    void setSolutionFieldT(const Tpetra_Vector& solnT);

    // Copy solution vector from Epetra_Vector into STK Mesh
    // Here soln is the local + neighbor (overlapped) solution
#if defined(ALBANY_EPETRA)
    void setOvlpSolutionField(const Epetra_Vector& soln);
#endif
    //Tpetra version of above
    void setOvlpSolutionFieldT(const Tpetra_Vector& solnT);

    int nonzeroesPerRow(const int neq) const;
    double monotonicTimeLabel(const double time);

    //! Return the maximum ID for the given entity type over the
    //! entire distributed mesh.  This method will perform a global
    //! reduction.
    stk::mesh::EntityId
    getMaximumID(const stk::mesh::EntityRank rank) const;

    //! Enrich the linear STK mesh to a spectral Albany mesh
    void enrichMeshLines();
    void enrichMeshQuads();

    //! Process spectral Albany mesh for owned nodal quantitites
    void computeOwnedNodesAndUnknownsLines();
    void computeOwnedNodesAndUnknownsQuads();

    //! Process coords for ML
    void setupMLCoords();

    //! Process spectral Albany mesh for overlap nodal quantitites
    void computeOverlapNodesAndUnknownsLines();
    void computeOverlapNodesAndUnknownsQuads();

    //! Fill in the Workset of coordinates with corner nodes from the
    //! STK mesh and enriched points from Gauss-Lobatto quadrature
    void computeCoordsLines();
    void computeCoordsQuads();

    //! Process spectral Albany mesh for CRS Graphs
    void computeGraphsLines();
    void computeGraphsQuads();

    //! Process spectral Albany mesh for Workset/Bucket Info
    void computeWorksetInfo();

    //! Process spectral Albany mesh for NodeSets
    void computeNodeSetsLines();

    //! Process spectral Albany mesh for SideSets
    void computeSideSetsLines();

    //! Create new STK mesh in which spectral elements are interpreted
    //! as a patch of linear quadrilaterals, and use this to setup
    //! Exodus output
    void createOutputMesh();
    
    void setupExodusOutput();

    //! Call stk_io for creating NetCDF output file
    void setupNetCDFOutput();
#if defined(ALBANY_EPETRA)
    int processNetCDFOutputRequest(const Epetra_Vector&);
#endif
    int processNetCDFOutputRequestT(const Tpetra_Vector&);

    //! Find the local side id number within parent element
    unsigned determine_local_side_id( const stk::mesh::Entity elem,
                                      stk::mesh::Entity side);
    //! Call stk_io for creating exodus output file
    Teuchos::RCP<Teuchos::FancyOStream> out;

    //! Convert the stk mesh on this processor to a nodal graph using SEACAS
    void meshToGraph();

    void writeCoordsToMatrixMarket() const;

    double previous_time_label;

  protected:

    Teuchos::RCP<Teuchos::ParameterList> discParams;

    //! Stk Mesh Objects
    stk::mesh::MetaData& metaData;
    stk::mesh::BulkData& bulkData;

    //! STK Mesh Struct for output
    Teuchos::RCP<Aeras::SpectralOutputSTKMeshStruct> outputStkMeshStruct;


#if defined(ALBANY_EPETRA)
    //! Epetra communicator
    Teuchos::RCP<const Epetra_Comm> comm;
#endif

    int spatial_dim; //how many spatial dimensions there are in the problem

    int points_per_edge; //number of points per edge (i.e., the degree of enrichment) -- read in from ParameterList.

    int nodes_per_element; //number of nodes of an element
  
    std::string element_name; //name of element

    //! Tpetra communicator and Kokkos node
    Teuchos::RCP<const Teuchos::Comm<int> > commT;

    //! Unknown map and node map
    Teuchos::RCP<const Tpetra_Map> node_mapT; 
    Teuchos::RCP<const Tpetra_Map> mapT; 

    //! Overlapped unknown map and node map
    Teuchos::RCP<const Tpetra_Map> overlap_mapT; 
    Teuchos::RCP<const Tpetra_Map> overlap_node_mapT; 

#if defined(ALBANY_EPETRA)
    Teuchos::RCP<Epetra_Map> node_map;
    Teuchos::RCP<Epetra_Map> map;
    Teuchos::RCP<Epetra_Map> overlap_node_map;
    Teuchos::RCP<Epetra_Map> overlap_map;

    NodalDOFsStructContainer nodalDOFsStructContainer;
#endif


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
    Albany::NodeSetGIDsList nodeSetGIDs;

    //! side sets stored as std::map(string ID, SideArray classes) per
    //! workset (std::vector across worksets)
    std::vector<Albany::SideSetList> sideSets;

    //! Flags indicating which edges are owned
    std::map< GO, bool > edgeIsOwned;

    //! Enriched edge map: GlobalOrdinal -> array of global node IDs
    std::map< GO, Teuchos::ArrayRCP< GO > > enrichedEdges;

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

    // States: vector of length worksets of a map from field name to shards array
    Albany::StateArrays stateArrays;
    std::vector<std::vector<std::vector<double> > > nodesOnElemStateVec;

    //! list of all owned nodes, saved for setting solution
    std::vector< stk::mesh::Entity > ownednodes ;
    std::vector< stk::mesh::Entity > cells ;

    //! list of all overlap nodes, saved for getting coordinates for mesh motion
    std::vector< stk::mesh::Entity > overlapnodes ;

    //! Number of elements on this processor
    int numOwnedNodes;
    int numOverlapNodes;
    GO numGlobalNodes;

    // Needed to pass coordinates to ML.
    Teuchos::RCP<Albany::RigidBodyModes> rigidBodyModes;

    int netCDFp;
    size_t netCDFOutputRequest;
    std::vector<int> varSolns;
    Albany::WorksetArray<Teuchos::ArrayRCP<std::vector<interp> > >::type interpolateData;

    // Storage used in periodic BCs to un-roll coordinates. Pointers
    // saved for destructor.
    std::vector<double*>  toDelete;

    Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct;

    // Used in Exodus writing capability
#ifdef ALBANY_SEACAS
    Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data;

    int outputInterval;

    size_t outputFileIdx;
#endif
    bool interleavedOrdering;

  private:

    Teuchos::RCP<Tpetra_CrsGraph> nodalGraph;


    // find the location of "value" within the first "count" locations of "vector"
    ssize_t in_list(const std::size_t value,
                    std::size_t count,
                    std::size_t *vector)
    {
      for(std::size_t i=0; i < count; i++)
      {
        if(vector[i] == value)
          return i;
      }
       return -1;
    }

    ssize_t in_list(const std::size_t value,
                    const Teuchos::Array<GO>& vector)
    {
      for (std::size_t i=0; i < vector.size(); i++)
        if (vector[i] == value)
          return i;
      return -1;
    }

    ssize_t in_list(const std::size_t value,
                    const std::vector<std::size_t>& vector)
    {
      for (std::size_t i=0; i < vector.size(); i++)
        if (vector[i] == value)
          return i;
      return -1;
    }

    ssize_t entity_in_list(const stk::mesh::Entity& value,
                           const std::vector<stk::mesh::Entity>& vec)
    {
      for (std::size_t i = 0; i < vec.size(); i++)
        if (bulkData.identifier(vec[i]) == bulkData.identifier(value))
          return i;
      return -1;
    }

    void printVertexConnectivity();

  };

}

#endif // AERAS_SPECTRALDISCRETIZATION_HPP
