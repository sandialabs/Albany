//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ASCII_STKMESH2DSTRUCT_HPP
#define ALBANY_ASCII_STKMESH2DSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"


// read ascii mesh and create corresponding STK mesh
// ascii file formats:

// Format: 0    #(this line is optional, if not present Format is assumed to be 0)
// Triangle 3   #shape (Triangle or Quadrilateral) and number of vertices (as of now it must be 3 for Triangle and 4 for Quadrilateral)
// 4 2 4        #[numVertices numElements numBoundaryEdges]
// 0.0 0.0 10   #[x y flag] for each vertex. flag is used to determine different node-sets.
// 1.0 0.0 10
// 1.0 1.0 20
// 0.0 1.0 30
// 1 2 3 0      #[Id node0 node1 node2 flag] for each element (quadrilaterals will have four nodes). flag is currently ignored
// 3 4 1 0
// 1 2 10       #(list of boundary edges and flag to determine different side sets)
// 2 3 10
// 3 4 20
// 4 1 20

// Format: 1        #(with format one an Id of each entity (vertices, elements, boundary edges) is provided))
// Triangle 3       #shape (Triangle or Quadrilateral) and number of vertices (as of now it must be 3 for Triangle and 4 for Quadrilateral)
// 4 2 4            #[numVertices numElements numBoundaryEdges]
// 100 0.0 0.0 10   #[Id x y flag], for each vertex. flag is used to determine different boundary node-sets
// 101 1.0 0.0 10
// 200 1.0 1.0 20
// 201 0.0 1.0 30
// 303 1 2 3 0      #[Id node0 node1 node2 flag] for each element. flag is currently ignored
// 304 3 4 1 0
// 901 1 2 10       #[Id node0 node1 flag] for each boundary edge. flag is used to determine different side sets
// 902 2 3 10
// 903 3 4 20
// 904 4 1 20

// For the above examples, the STK mesh will have a "node_set" containing all the vertices,
// "boundary_node_set_10", "boundary_node_set_20" and "boundary_node_set_30" containing the boundary nodes with flag 10, 20 and 30, respectively,
// "boundary_side_set" containing all the boundary edges,
// "boundary_edge_set_10" and "boundary_edge_set_20" containing the boundary edges with flag 10 and 20, respectively


namespace Albany {

  class AsciiSTKMesh2D : public GenericSTKMeshStruct {

    public:

    AsciiSTKMesh2D(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
		  const int numParams);

    ~AsciiSTKMesh2D();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
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
    //Ioss::Init::Initializer ioInit;

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParameters() const;

    bool periodic;
    int NumElemNodes; //number of nodes per element (e.g. 3 for Triangles)
    int NumNodes; //number of nodes
    int NumElems; //number of elements
    int NumBdEdges; //number of faces on basal boundary
    std::map<int,std::string> bdTagToNodeSetName;
    std::map<int,std::string> bdTagToSideSetName;
    std::vector<int> coord_Ids, ele_Ids, be_Ids;
    std::vector<int> coord_flags;
    std::vector<std::vector<double>> coords;
    std::vector<std::vector<int>> elems;
    std::vector<std::vector<int>> bdEdges;
  };

}
#endif
