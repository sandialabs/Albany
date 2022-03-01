//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef EVAL_TESTSETUP_HPP
#define EVAL_TESTSETUP_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "PHAL_ConvertFieldType.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"

#include "Shards_CellTopology.hpp"
#include "Albany_ProblemUtils.hpp"
#include "PHAL_ComputeBasisFunctions.hpp"


namespace Albany {

template<typename EvalType, typename Traits>
Teuchos::RCP<PHAL::ComputeBasisFunctions<EvalType, Traits>>
createTestLayoutAndBasis(Teuchos::RCP<Albany::Layouts> &dl, 
         const int worksetSize, const int cubatureDegree, const int numDim)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using std::string;
   using PHAL::AlbanyTraits;

   const CellTopologyData *elem_top = shards::getCellTopologyData<shards::Hexahedron<8> >();

   RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis = 
       Albany::getIntrepid2Basis(*elem_top);

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (elem_top));

   const int numNodes = intrepidBasis->getCardinality();

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device> > cellCubature = 
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, cubatureDegree);

   const int numQPtsCell = cellCubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

   std::cout << "Field Dimensions: Workset=" << worksetSize
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPtsCell
        << ", Dim= " << numDim << std::endl;

   dl = rcp(new Albany::Layouts(worksetSize, numVertices, numNodes, numQPtsCell, numDim));

   if(dl->vectorAndGradientLayoutsAreEquivalent == false){
      std::cout << "In Data Layout vecDim != numDim" << std::endl;
   }

   RCP<ParameterList> bfp = rcp(new ParameterList("Compute Basis Functions"));

   // Inputs: X, Y at nodes, Cubature, and Basis
   bfp->set<string>("Coordinate Vector Name", coord_vec_name);
   bfp->set< RCP<Intrepid2::Cubature<PHX::Device> > >("Cubature", cellCubature);

   bfp->set< RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > >
       ("Intrepid2 Basis", intrepidBasis);

   bfp->set<RCP<shards::CellTopology> >("Cell Type", cellType);

   // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
   bfp->set<std::string>("Weights Name",              Albany::weights_name);
   bfp->set<std::string>("Jacobian Det Name",         Albany::jacobian_det_name);
   bfp->set<std::string>("Jacobian Name",             Albany::jacobian_det_name);
   bfp->set<std::string>("Jacobian Inv Name",         Albany::jacobian_inv_name);
   bfp->set<std::string>("BF Name",                   Albany::bf_name);
   bfp->set<std::string>("Weighted BF Name",          Albany::weighted_bf_name);
   bfp->set<std::string>("Gradient BF Name",          Albany::grad_bf_name);
   bfp->set<std::string>("Weighted Gradient BF Name", Albany::weighted_grad_bf_name);

   return Teuchos::rcp(new PHAL::ComputeBasisFunctions<EvalType,Traits>(*bfp, dl));

}

void createTestMapsAndWorksetConns(
    Teuchos::RCP<Tpetra_Map> &cell_map,
    Teuchos::RCP<Tpetra_Map> &overlapped_node_map,
    Teuchos::RCP<Tpetra_Map> &overlapped_dof_map,
    Albany::WorksetConn wsGlobalElNodeEqID,
    Albany::WorksetConn wsLocalElNodeEqID,
    unsigned int numCells_per_direction,
    unsigned int nodes_per_element,
    unsigned int neq,
    Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
  const unsigned int numCells_per_layer = numCells_per_direction * numCells_per_direction;
  const unsigned int numCells = numCells_per_layer * numCells_per_direction;
  const unsigned int numNodes_per_layer = (numCells_per_direction + 1) * (numCells_per_direction + 1);
  const unsigned int numNodes = numNodes_per_layer * (numCells_per_direction + 1);

  std::vector<unsigned int> node_offset = {0, 1, numCells_per_direction + 2, numCells_per_direction + 1, numNodes_per_layer, numNodes_per_layer + 1, numNodes_per_layer + numCells_per_direction + 2, numNodes_per_layer + numCells_per_direction + 1};

  // This numbering follows Hex_QP_Numbering.pdf
  for (unsigned int i_z = 0; i_z < numCells_per_direction; ++i_z)
  {
    for (unsigned int i_y = 0; i_y < numCells_per_direction; ++i_y)
    {
      for (unsigned int i_x = 0; i_x < numCells_per_direction; ++i_x)
      {
        const unsigned int cell_id = i_z * numCells_per_layer + i_y * numCells_per_direction + i_x;
        const unsigned int first_node_id = i_z * numNodes_per_layer + i_y * (numCells_per_direction + 1) + i_x;
        for (unsigned int node = 0; node < nodes_per_element; ++node)
        {
          for (unsigned int eq = 0; eq < neq; ++eq)
          {
            wsGlobalElNodeEqID(cell_id, node, eq) = (first_node_id + node_offset[node]) * neq + eq;
          }
        }
      }
    }
  }

  cell_map = Teuchos::rcp(new Tpetra_Map(numCells, 0, comm));

  std::vector<Tpetra_GO> overlapped_nodes = {};
  std::vector<Tpetra_GO> overlapped_dofs = {};

  for (unsigned int cell = 0; cell < cell_map->getLocalNumElements(); ++cell)
    for (unsigned int node = 0; node < nodes_per_element; ++node)
    {
      const Tpetra_GO nodeID = wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, 0) / neq;
      if (std::find(overlapped_nodes.begin(),
                    overlapped_nodes.end(),
                    nodeID) == overlapped_nodes.end())
      {
        overlapped_nodes.push_back(nodeID);
        for (unsigned int eq = 0; eq < neq; ++eq)
          overlapped_dofs.push_back(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, eq));
      }
    }

  overlapped_node_map = Teuchos::rcp(new Tpetra_Map(numNodes, overlapped_nodes, 0, comm));
  overlapped_dof_map = Teuchos::rcp(new Tpetra_Map(numNodes * neq, overlapped_dofs, 0, comm));

  for (unsigned int cell = 0; cell < cell_map->getLocalNumElements(); ++cell)
    for (unsigned int node = 0; node < nodes_per_element; ++node)
      for (unsigned int eq = 0; eq < neq; ++eq)
        wsLocalElNodeEqID(cell, node, eq) = overlapped_dof_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell), node, eq));
}

}


#endif // EVAL_TESTSETUP
