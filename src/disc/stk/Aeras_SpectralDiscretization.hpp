//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
      const int points_per_edge,  
      const Teuchos::RCP<Teuchos::ParameterList>& discParams)
    {
      Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream(); 
#ifdef OUTPUT_TO_SCREEN
      *out << "DEBUG: in AerasMeshSpectStruct!  Element Degree =  "
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
      *out << "DEBUG: original ctd name = " << orig_name << std::endl; 
      *out << "DEBUG: original ctd key = " << orig_ctd.key << std::endl; 
#endif 
      int orig_numDim = orig_mesh_specs_struct->numDim;
      //int orig_cubatureDegree = orig_mesh_specs_struct->cubatureDegree;
      int new_cubatureDegree = setCubatureDegree(points_per_edge-1, discParams); 
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
      std::string cub = discParams->get("Cubature Rule", "GAUSS_LOBATTO"); 
      //If cubature rule in input file is not GAUSS_LOBATTO, print warning and reset cubature 
      //to Gauss-Lobatto.
      if (cub != "GAUSS_LOBATTO") 
         *out << "Setting Cubature Rule to GAUSS_LOBATTO. \n"; 
      else 
          *out << "Using Cubature Rule specified in input file: GAUSS_LOBATTO. \n";  
      
      const Intrepid2::EPolyType new_cubatureRule 
          = static_cast<Intrepid2::EPolyType>(Intrepid2::POLYTYPE_GAUSS_LOBATTO);

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
      //For 1D elements, create a new key for the ctd -- this is needed for Intrepid2
      //setJacobian function. 
      if (orig_numDim == 1) 
        new_ctd.key = shards::cellTopologyKey(orig_numDim, 0, 0, 2, np); 
#ifdef OUTPUT_TO_SCREEN
      *out << "DEBUG: new_ctd.name = " << new_ctd.name << std::endl; 
      *out << "DEBUG: new_ctd.key = " << new_ctd.key << std::endl; 
#endif
      // Create and return Albany::MeshSpecsStruct object based on the
      // new (enriched) ctd.
      return Teuchos::rcp(new Albany::MeshSpecsStruct(new_ctd,
                                                      orig_numDim,
                                                      new_cubatureDegree,
                                                      orig_nsNames,
                                                      orig_ssNames,
                                                      orig_worksetSize,
                                                      orig_ebName,
                                                      orig_ebNameToIndex,
                                                      orig_interleavedOrdering,
                                                      orig_sepEvalsByEB,
                                                      new_cubatureRule));
      delete [] new_name_char;
    }

    //The following function sets the cubature degree based on the element degree for spectral elements, 
    //so that the user does not need to worry about specifying this in the input file. 
    //Cubature rules are only implemented for elements up to degree 12.  Cubature rules for 
    //higher order elements may be added, if desired. 
    int setCubatureDegree(
      const int elementDegree,  
      const Teuchos::RCP<Teuchos::ParameterList>& discParams) 
    {
      Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream(); 
      int cubatureDegree; 
      switch (elementDegree)
      {
        case 1: cubatureDegree = 1; break;
        case 2: cubatureDegree = 3; break; 
        case 3: cubatureDegree = 4; break;
        case 4: cubatureDegree = 6; break; 
        case 5: cubatureDegree = 8; break;
        case 6: cubatureDegree = 10; break; 
        case 7: cubatureDegree = 12; break; 
        case 8: cubatureDegree = 14; break; 
        case 9: cubatureDegree = 16; break; 
        case 10: cubatureDegree = 18; break; 
        case 11: cubatureDegree = 20; break; 
        case 12: cubatureDegree = 22; break;  
        default:
           TEUCHOS_TEST_FOR_EXCEPTION(
              true, std::logic_error,
             "Cubature Degree is not implemented for element of degree "<< elementDegree << "!  " <<
             "To use an element of this order, please implement the right cubatureDegree to the setCubature " << 
             "function in Aeras_SpectralDiscretization.hpp.");
      }
      int orig_cubatureDegree = discParams->get("Cubature Degree", 3);
      if (orig_cubatureDegree == cubatureDegree) 
         *out << "Setting Cubature Degree to default value or value specified in input file: " << orig_cubatureDegree << std::endl;  
      *out << "Setting Cubature Degree to " << cubatureDegree << " for element of degree " << elementDegree << std::endl; 
      return cubatureDegree; 
    }
  };

  class SpectralDiscretization : public Albany::AbstractDiscretization
  {
  public:

    //! Constructor
    SpectralDiscretization(
       const Teuchos::RCP<Teuchos::ParameterList>& discParams,
       Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct,
       const int numTracers, 
       const int numLevels, 
       const Teuchos::RCP<const Teuchos_Comm>& commT,
       const bool explicit_scheme, 
       const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes=Teuchos::null);

    //! Destructor
    ~SpectralDiscretization();

#if defined(ALBANY_EPETRA)
    //! Get Epetra DOF map
    Teuchos::RCP<const Epetra_Map> getMap() const;

    //! Get overlapped node map
    Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const;
#endif

    //! Get Tpetra DOF map
    Teuchos::RCP<const Tpetra_Map> getMapT() const;

    //! Get Tpetra overlapped DOF map
    Teuchos::RCP<const Tpetra_Map> getOverlapMapT() const;

    //! Get field overlapped node map
    Teuchos::RCP<const Tpetra_Map> getOverlapNodeMapT(const std::string& field_name) const;

    //! Get field DOF map
    Teuchos::RCP<const Tpetra_Map> getMapT(const std::string& field_name) const;

#if defined(ALBANY_EPETRA)
    //! Get Epetra Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const;
#endif
    //! Get Tpetra Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getJacobianGraphT() const;
    
    //! Get Tpetra implicit Jacobian graph (non-diagonal) 
    Teuchos::RCP<const Tpetra_CrsGraph> getImplicitJacobianGraphT() const;
    
#if defined(ALBANY_EPETRA)
    //! Get Epetra overlap Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const;
#endif
    //! Get Tpetra overlap Jacobian graph
    Teuchos::RCP<const Tpetra_CrsGraph> getOverlapJacobianGraphT() const;
    
    //! Get Tpetra overlap implicit Jacobian graph (non-diagonal) 
    Teuchos::RCP<const Tpetra_CrsGraph> getImplicitOverlapJacobianGraphT() const;

#if defined(ALBANY_EPETRA)
    //! Get field node map
    Teuchos::RCP<const Epetra_Map> getNodeMap() const;
    //! Get field overlapped DOF map
    Teuchos::RCP<const Epetra_Map> getOverlapMap() const;
#endif

    //! Get Tpetra Node map
    Teuchos::RCP<const Tpetra_Map> getNodeMapT() const;
    //! Get field Tpetra node map
    Teuchos::RCP<const Tpetra_Map> getNodeMapT(const std::string& field_name) const;
    //! Get field overlapped DOF map
    Teuchos::RCP<const Tpetra_Map> getOverlapNodeMapT() const;
    //! Get field overlapped DOF map
    Teuchos::RCP<const Tpetra_Map> getOverlapMapT(const std::string& field_name) const;

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
    const Albany::WsLIDList& getElemGIDws() const
    {
      return elemGIDws;
    };

    //! Get map from (Ws, El, Local Node) -> NodeLID
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type&
    getWsElNodeEqID() const;

    //! Get map from (Ws, Local Node) -> NodeGID
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    getWsElNodeID() const;


    //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID,
    //! works for both scalar and vector fields
    const std::vector<Albany::IDArray>&
    getElNodeEqID(const std::string& field_name) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::SpectralDiscretization: getElNodeEqID(const std::string& field_name) const not implemented");
    }

    const Albany::NodalDOFManager&
    getDOFManager(const std::string& field_name) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::SpectralDiscretization: getDOFManager(const std::string& field_name) const not implemented");
    }

    const Albany::NodalDOFManager&
    getOverlapDOFManager(const std::string& field_name) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Albany::SpectralDiscretization: getOverlapDOFManager(const std::string& field_name) const not implemented");

    }

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

    const Albany::WorksetArray<Teuchos::ArrayRCP<double*> >::type&
    getLatticeOrientation() const;

    //! Print the coordinates for debugging
    void printCoords() const;
    void printConnectivity(bool printEdges=false) const;
    void printCoordsAndGIDs() const;

    //! Get sideSet discretizations map
    const SideSetDiscretizationsType& getSideSetDiscretizations () const
    {
      //Warning, returning an empty sideSetDiscretizations. 
      return sideSetDiscretizations;
    }

    //! Get the map side_id->side_set_elem_id
    const std::map<std::string,std::map<GO,GO> >& getSideToSideSetCellMap () const
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Functionality not supported by Aeras discretization.\n");
      return sideToSideSetCellMap;
    }

    //! Get the map side_node_id->side_set_cell_node_id
    const std::map<std::string,std::map<GO,std::vector<int>>>& getSideNodeNumerationMap () const
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Functionality not supported by Aeras discretization.\n");
      return sideNodeNumerationMap;
    }

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
   void writeSolutionMV(const Tpetra_MultiVector& solnT,
                       const double time,
                       const bool overlapped = false);

   void writeSolutionToMeshDatabaseT(const Tpetra_Vector &solutionT,
                                     const double time,
                                     const bool overlapped = false);
   void writeSolutionMVToMeshDatabase(const Tpetra_MultiVector &solutionT,
                                     const double time,
                                     const bool overlapped = false);

   void writeSolutionToFileT(const Tpetra_Vector& solnT,
                             const double time,
                             const bool overlapped = false);
   void writeSolutionMVToFile(const Tpetra_MultiVector& solnT,
                             const double time,
                             const bool overlapped = false);

#if defined(ALBANY_EPETRA)
    Teuchos::RCP<Epetra_Vector>
    getSolutionField(const bool overlapped=false) const;
#endif
    //Tpetra analog
    Teuchos::RCP<Tpetra_Vector>
    getSolutionFieldT(const bool overlapped=false) const;

    Teuchos::RCP<Tpetra_MultiVector>
    getSolutionMV(const bool overlapped=false) const;

    //Tpetra analog
    void setResidualFieldT(const Tpetra_Vector& residualT);

    //Retrieve mesh struct
    Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() const {return stkMeshStruct;}

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

    //! Spectral supports MOR
    virtual bool supportsMOR() const
    {
      return false;
    }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const
    {
      return stkMeshStruct->restartDataTime();
    }

    //! After mesh modification, need to update the element
    //! connectivity and nodal coordinates
    void updateMesh();

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
   
    bool isExplicitScheme() const 
    {
      return explicit_scheme; 
    }

    //! Get number of levels (for hydrostatic problems) 
    int getNumLevels() const { return numLevels; }

    //! Get number of tracers (for hydrostatic problems) 
    int getNumTracers() const { return numTracers; }

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

    void getSolutionMV(Tpetra_MultiVector &resultT,
                           bool overlapped=false) const;

    //! Copy field from STK Mesh field to given Epetra_Vector
    void getFieldT(Tpetra_Vector &field_vector,
                  const std::string& field_name) const;

    // Copy field vector into STK Mesh field
    void setFieldT(const Tpetra_Vector &field_vector,
                  const std::string& field_name,
                  bool overlapped=false);

    //Tpetra version of above
    void setSolutionFieldT(const Tpetra_Vector& solnT);
    void setSolutionFieldMV(const Tpetra_MultiVector& solnT);

    // Copy solution vector from Epetra_Vector into STK Mesh
    // Here soln is the local + neighbor (overlapped) solution
    void setOvlpSolutionFieldT(const Tpetra_Vector& solnT);
    void setOvlpSolutionFieldMV(const Tpetra_MultiVector& solnT);

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
    Teuchos::RCP<Tpetra_CrsGraph> computeOverlapGraph();
    Teuchos::RCP<Tpetra_CrsGraph> computeOwnedGraph(Teuchos::RCP<Tpetra_CrsGraph> overlap_graphT_);

    //  The following function allocates the graph of a diagonal Jacobian, 
    //  relevant for explicit schemes.
    void computeGraphs_Explicit();

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

    int processNetCDFOutputRequestT(const Tpetra_Vector&);

    //! Find the local side id number within parent element
    unsigned determine_local_side_id( const stk::mesh::Entity elem,
                                      stk::mesh::Entity side);
    //! Call stk_io for creating exodus output file
    Teuchos::RCP<Teuchos::FancyOStream> out;

    void writeCoordsToMatrixMarket() const;

    double previous_time_label;

    //Create enum type for the different kinds of elements (currently lines and quads)
    enum elemType {LINE, QUAD};
    elemType ElemType;

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

    //! Jacobian matrix graph
    Teuchos::RCP<Tpetra_CrsGraph> graphT; 
    
    //! Jacobian matrix implicit graph
    Teuchos::RCP<Tpetra_CrsGraph> implicit_graphT; 

    //! Overlapped Jacobian matrix graph
    Teuchos::RCP<Tpetra_CrsGraph> overlap_graphT; 
    
    //! Overlapped Jacobian matrix implicit graph
    Teuchos::RCP<Tpetra_CrsGraph> implicit_overlap_graphT; 

    //! Processor ID
    unsigned int myPID;

    //! Number of equations (and unknowns) per node
    const unsigned int neq;

    //! Number of levels (for hydrostatic equations) 
    const int numLevels; 
    
    //! Flag for explicit scheme
    const bool explicit_scheme; 
    
    //! number of tracers (for hydristatic equations) 
    const int numTracers; 

    //! Number of elements on this processor
    unsigned int numMyElements;

    //! node sets stored as std::map(string ID, int vector of GIDs)
    Albany::NodeSetList nodeSets;
    Albany::NodeSetCoordList nodeSetCoords;
    Albany::NodeSetGIDsList nodeSetGIDs;

    //! side sets stored as std::map(string ID, SideArray classes) per
    //! workset (std::vector across worksets)
    std::vector<Albany::SideSetList> sideSets;

    // Side set discretizations related structures (not supported but needed for getters return values)
    std::map<std::string,Teuchos::RCP<Albany::AbstractDiscretization> > sideSetDiscretizations;
    std::map<std::string,std::map<GO,GO> >                              sideToSideSetCellMap;
    std::map<std::string,std::map<GO,std::vector<int> > >               sideNodeNumerationMap;

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
    Albany::WorksetArray<Teuchos::ArrayRCP<double*> >::type latticeOrientation;

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

  };

}

#endif // AERAS_SPECTRALDISCRETIZATION_HPP
