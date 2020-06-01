//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GMSH_STK_MESH_STRUCT_HPP
#define ALBANY_GMSH_STK_MESH_STRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

//#include <Ionit_Initializer.h>

namespace Albany
{

enum class GmshVersion
{
  V2_2,
  V4_1
};

class GmshSTKMeshStruct : public GenericSTKMeshStruct
{
  public:

  GmshSTKMeshStruct (const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<const Teuchos_Comm>& commT,
		     const int numParams);

  ~GmshSTKMeshStruct();

  void setFieldAndBulkData (const Teuchos::RCP<const Teuchos_Comm>& commT,
                            const Teuchos::RCP<Teuchos::ParameterList>& params,
                            const unsigned int neq_,
                            const AbstractFieldContainer::FieldContainerRequirements& req,
                            const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                            const unsigned int worksetSize,
                            const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {},
                            const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req = {});

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const {return false; }

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const {return -1.0; }

  private:

  // Set boundary information.
  // Includes sideset and nodeset names and counts.
  void set_boundaries( const Teuchos::RCP<const Teuchos_Comm>& commT,
                       std::vector<std::string>&               ssNames,
                       std::vector<std::string>&               nsNames);

  // Check the version of the input msh file
  void check_version( std::ifstream& ifile);

  // Sets the set of allowable gmsh versions; i.e., 
  // versions we know how to read
  void set_allowable_gmsh_versions();

  // Sets the version enum from the float read from the mesh file.
  // Returns false if the version cannot be read.
  bool set_version_enum_from_float();

  // Looks through ifile for the line containing the
  // line_of_interest.
  void swallow_lines_until( std::ifstream& ifile, 
                            std::string&   line, 
                            std::string    line_of_interest);

  Teuchos::RCP<const Teuchos::ParameterList> getValidDiscretizationParameters() const;

  // Gets the physical name-tag pairs for version 4.1 meshes
  void get_physical_names( std::map<std::string, int>&             physical_names,
                           const Teuchos::RCP<const Teuchos_Comm>& commT);

  // Share physical_names map with all other proccesses
  void broadcast_physical_names( std::map<std::string, int>&             physical_names,
                                 const Teuchos::RCP<const Teuchos_Comm>& commT);

  // Read the physical names for Gmsh V 4.1 
  // to populate the physical_names map
  void read_physical_names_from_file( std::map<std::string, int>& physical_names);

  // Opens the gmsh msh file. Variable `fname` must be set.
  // Don't forget to close when done!
  // Uses Teuchos test to check if file is open.
  void open_fname( std::ifstream& ifile);

  // Determine the type of the msh file
  void determine_file_type( bool& legacy, bool& binary, bool& ascii);

  // Broadcast topology of the mesh from 0 to all over procs
  void broadcast_topology( const Teuchos::RCP<const Teuchos_Comm>& commT);

  // Sets NumNodes for ascii msh files
  void set_NumNodes( std::ifstream& ifile);

  // Reads in the node data
  void load_node_data( std::ifstream& ifile);

  // Sets the number of entities, elements and cells
  void set_num_entities( std::ifstream& ifile);

  // Sets the number of each type of element
  void set_specific_num_of_each_elements( std::ifstream& ifile);

  // Increments the element type counter based on the type number
  void increment_element_type( int e_type);

  // Allocates memory for element pointers below
  void size_all_element_pointers();

  // Set mesh info like dimension, number of elements, sides, etc.
  void set_generic_mesh_info();

  // Reads in the element info.
  // Includes tags and nodes belonging to each element.
  void load_element_data( std::ifstream& ifile);

  // Stores element info based on e_type. Updates i(type) counters.
  // Records tags for the element.
  void store_element_info( int  e_type,
                           int& iline,
                           int& iline3,
                           int& itria,
                           int& itri6,
                           int& iquad,
                           int& itetra,
                           int& itet10,
                           int& ihexa,
                           std::vector<int>& tags,
                           std::stringstream& ss);

  // Create the element blocks
  // Current only creates `Element Block 0` 
  void create_element_block();

  // Creates a nodeset will all nodes
  void set_all_nodes_boundary( std::vector<std::string>& nsNames);

  // Creates a sideset with all sides
  void set_all_sides_boundary( std::vector<std::string>& ssNames);

  // Broadcast a single name-tag pair from proc 0 to all others
  void broadcast_name_tag_pair( std::vector< std::string>               names,
                                int*                                    tags_array,
                                int                                     pair_number,
                                const Teuchos::RCP<const Teuchos_Comm>& commT,
                                std::map< std::string, int>&            physical_names);

  // Reads a single physical name from ifile.
  // Prepends with an undescore and remove quotation marks.
  void get_name_for_physical_names( std::string& name, std::ifstream& ifile);

  // Reads ifile to map surface tags to physical tags.
  // Reports error if any surface is associted with more than one tag.
  void get_physical_tag_to_surface_tag_map( std::ifstream&      ifile, 
                                            std::map<int, int>& physical_surface_tags,
                                            int                 num_surfaces);

  
  // Adds a sideset with name sideset_name and side tag number tag.
  void add_sideset( std::string sideset_name, int tag, std::vector<std::string>& ssNames);

  // Adds a nodeset with name nodeset_name and node tag number tag.
  void add_nodeset( std::string nodeset_name, int tag, std::vector<std::string>& nsNames);

  // The version of the gmsh msh file
  GmshVersion version;

  // The float of the version read from the msh file
  float version_in;

  // The file name of the msh file
  std::string fname;

  // The set of versions we know how to read
  std::set<float> allowable_gmsh_versions;

  // Map from element block names to their index
  std::map<std::string,int> ebNameToIndex;

  void loadLegacyMesh ();
  void loadAsciiMesh ();
  void loadBinaryMesh ();


  // Init the int counters below to zero.
  void init_counters_to_zero();

  // Init the int pointers below to null.
  void init_pointers_to_null();


  // The number of entities, both elements and cells
  int num_entities;

  int NumElemNodes; // Number of nodes per element (e.g. 3 for Triangles)
  int NumSideNodes; // Number of nodes per side (e.g. 2 for a Line)
  int NumNodes; //number of nodes
  int NumElems; //number of elements
  int NumSides; //number of sides

  std::map<int,std::string> bdTagToNodeSetName;
  std::map<int,std::string> bdTagToSideSetName;
  double (*pts)[3];

  // Only some will be used, but it's easier to have different pointers
  int** hexas;
  int** tetra;
  int** tet10;
  int** quads;
  int** trias;
  int** tri6;
  int** lines;
  int** line3;

  // Number of each of the above element types
  int nb_hexas;
  int nb_tetra;
  int nb_tet10;
  int nb_quads;
  int nb_trias;
  int nb_line3;
  int nb_tri6;
  int nb_lines;

  // These pointers will be set equal to two of the previous group, depending on dimension
  // NOTE: do not call delete on these pointers! Delete the previous ones only!
  int** elems;
  int** sides;
};

} // Namespace Albany

#endif // ALBANY_GMSH_STK_MESH_STRUCT_HPP
