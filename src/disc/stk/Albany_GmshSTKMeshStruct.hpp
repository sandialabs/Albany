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
  std::ifstream open_fname();

  // Sets NumNodes for ascii msh files
  void set_NumNodes( std::ifstream& ifile);

  // Reads in the node data
  void load_node_data( std::ifstream& ifile);

  // The version of the gmsh msh file
  float version;

  // The file name of the msh file
  std::string fname;

  // The set of versions we know how to read
  std::set<float> allowable_gmsh_versions;

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

  // These pointers will be set equal to two of the previous group, depending on dimension
  // NOTE: do not call delete on these pointers! Delete the previous ones only!
  int** elems;
  int** sides;
};

} // Namespace Albany

#endif // ALBANY_GMSH_STK_MESH_STRUCT_HPP
