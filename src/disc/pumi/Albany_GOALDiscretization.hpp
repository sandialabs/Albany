//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GOALDISCRETIZATION_HPP
#define ALBANY_GOALDISCRETIZATION_HPP

#include "Albany_PUMIDiscretization.hpp"
#include "Albany_GOALMeshStruct.hpp"

namespace Albany {

struct GOALNode
{
  int lid;
  bool higherOrder;
};

typedef std::map<std::string, std::vector<GOALNode> > GOALNodeSets;

class GOALDiscretization : public PUMIDiscretization
{
  public:

    //! Constructor
    GOALDiscretization(
       Teuchos::RCP<Albany::GOALMeshStruct> goalMeshStruct,
       const Teuchos::RCP<const Teuchos_Comm>& commT,
       const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes = Teuchos::null);

    //! Destructor
    ~GOALDiscretization();

    //! Retrieve mesh struct
    Teuchos::RCP<Albany::GOALMeshStruct> getGOALMeshStruct() {return goalMeshStruct;}

    //! Get the coordinate vector
    const Teuchos::ArrayRCP<double>& getCoordinates() const;

    //! Setup coordinates for MueLu
    void setupMLCoords();

    //! Retrieve the goalNodeSets
    GOALNodeSets getGOALNodeSets() {return goalNodeSets;}

    //! Get the number of DOFs per element for this element block
    int getNumNodesPerElem(int ebi);

    //! Attach the solution to the APF mesh from a Tpetra vector
    void attachSolutionToMesh(Tpetra_Vector const& x);

    //! Fill a solution vector from the mesh
    void fillSolutionVector(Teuchos::RCP<Tpetra_Vector>& x);

    //! Update the mesh
    void updateMesh(bool shouldTransferIPData);

  private:

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

    //! Goal node sets
    GOALNodeSets goalNodeSets;

    //! Goal mesh struct
    Teuchos::RCP<Albany::GOALMeshStruct> goalMeshStruct;

    //! Vertex information
    int numOverlapVertices;
    apf::Numbering* vtxNumbering;
    apf::DynamicArray<apf::Node> vertices;
};

}

#endif
