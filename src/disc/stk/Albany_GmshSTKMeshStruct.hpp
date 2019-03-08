//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GMSH_STK_MESH_STRUCT_HPP
#define ALBANY_GMSH_STK_MESH_STRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"

//#include <Ionit_Initializer.h>

namespace Albany
{

class GmshSTKMeshStruct : public GenericSTKMeshStruct
{
  public:

  GmshSTKMeshStruct (const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<const Teuchos_Comm>& commT);

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

  // Looks through ifile for the line containing the
  // line_of_interest.
  void swallow_lines_until( std::ifstream& ifile, 
                            std::string&   line, 
                            std::string    line_of_interest);

  Teuchos::RCP<const Teuchos::ParameterList> getValidDiscretizationParameters() const;

  // Gets the physical name-tag pairs for version 4.1 meshes
  void get_physical_names( std::map<std::string, int>&  physical_names);

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
                           int& itria,
                           int& iquad,
                           int& itetra,
                           int& ihexa,
                           std::vector<int>& tags,
                           std::stringstream& ss);

  // Create the element blocks
  // Current only creates `Element Block 0` 
  void create_element_block();

  // The version of the gmsh msh file
  float version;

  // The file name of the msh file
  std::string fname;

  // The set of versions we know how to read
  std::set<float> allowable_gmsh_versions;

  // Map from element block names to their index
  std::map<std::string,int> ebNameToIndex;

  void loadLegacyMesh ();
  void loadAsciiMesh ();
  void loadBinaryMesh ();

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
  int** quads;
  int** trias;
  int** lines;

  // Number of each of the above element types
  int nb_hexas;
  int nb_tetra;
  int nb_quads;
  int nb_trias;
  int nb_lines;

  // These pointers will be set equal to two of the previous group, depending on dimension
  // NOTE: do not call delete on these pointers! Delete the previous ones only!
  int** elems;
  int** sides;
};

} // Namespace Albany

#endif // ALBANY_GMSH_STK_MESH_STRUCT_HPP
