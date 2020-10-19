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

Albany::WorksetConn createTestMapsAndWorksetConns(
   Teuchos::RCP<Tpetra_Map> &cell_map,
   Teuchos::RCP<Tpetra_Map> &overlapped_node_map,
   Albany::WorksetConn wsGlobalElNodeEqID,
   int numCells,
   int nodes_per_element,
   int neq,
   int x_size,
   Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
   wsGlobalElNodeEqID(0,0,0) = 0;
   wsGlobalElNodeEqID(0,1,0) = 1;
   wsGlobalElNodeEqID(0,2,0) = 4;
   wsGlobalElNodeEqID(0,3,0) = 3;
   wsGlobalElNodeEqID(0,4,0) = 9;
   wsGlobalElNodeEqID(0,5,0) = 10;
   wsGlobalElNodeEqID(0,6,0) = 13;
   wsGlobalElNodeEqID(0,7,0) = 12;

   wsGlobalElNodeEqID(1,0,0) = 1;
   wsGlobalElNodeEqID(1,1,0) = 2;
   wsGlobalElNodeEqID(1,2,0) = 5;
   wsGlobalElNodeEqID(1,3,0) = 4;
   wsGlobalElNodeEqID(1,4,0) = 10;
   wsGlobalElNodeEqID(1,5,0) = 11;
   wsGlobalElNodeEqID(1,6,0) = 14;
   wsGlobalElNodeEqID(1,7,0) = 13;

   wsGlobalElNodeEqID(2,0,0) = 3;
   wsGlobalElNodeEqID(2,1,0) = 4;
   wsGlobalElNodeEqID(2,2,0) = 7;
   wsGlobalElNodeEqID(2,3,0) = 6;
   wsGlobalElNodeEqID(2,4,0) = 12;
   wsGlobalElNodeEqID(2,5,0) = 13;
   wsGlobalElNodeEqID(2,6,0) = 16;
   wsGlobalElNodeEqID(2,7,0) = 15;

   wsGlobalElNodeEqID(3,0,0) = 4;
   wsGlobalElNodeEqID(3,1,0) = 5;
   wsGlobalElNodeEqID(3,2,0) = 8;
   wsGlobalElNodeEqID(3,3,0) = 7;
   wsGlobalElNodeEqID(3,4,0) = 13;
   wsGlobalElNodeEqID(3,5,0) = 14;
   wsGlobalElNodeEqID(3,6,0) = 17;
   wsGlobalElNodeEqID(3,7,0) = 16;

   wsGlobalElNodeEqID(4,0,0) = 9;
   wsGlobalElNodeEqID(4,1,0) = 10;
   wsGlobalElNodeEqID(4,2,0) = 13;
   wsGlobalElNodeEqID(4,3,0) = 12;
   wsGlobalElNodeEqID(4,4,0) = 18;
   wsGlobalElNodeEqID(4,5,0) = 19;
   wsGlobalElNodeEqID(4,6,0) = 22;
   wsGlobalElNodeEqID(4,7,0) = 21;

   wsGlobalElNodeEqID(5,0,0) = 10;
   wsGlobalElNodeEqID(5,1,0) = 11;
   wsGlobalElNodeEqID(5,2,0) = 14;
   wsGlobalElNodeEqID(5,3,0) = 13;
   wsGlobalElNodeEqID(5,4,0) = 19;
   wsGlobalElNodeEqID(5,5,0) = 20;
   wsGlobalElNodeEqID(5,6,0) = 23;
   wsGlobalElNodeEqID(5,7,0) = 22;

   wsGlobalElNodeEqID(6,0,0) = 12;
   wsGlobalElNodeEqID(6,1,0) = 13;
   wsGlobalElNodeEqID(6,2,0) = 16;
   wsGlobalElNodeEqID(6,3,0) = 15;
   wsGlobalElNodeEqID(6,4,0) = 21;
   wsGlobalElNodeEqID(6,5,0) = 22;
   wsGlobalElNodeEqID(6,6,0) = 25;
   wsGlobalElNodeEqID(6,7,0) = 24;

   wsGlobalElNodeEqID(7,0,0) = 13;
   wsGlobalElNodeEqID(7,1,0) = 14;
   wsGlobalElNodeEqID(7,2,0) = 17;
   wsGlobalElNodeEqID(7,3,0) = 16;
   wsGlobalElNodeEqID(7,4,0) = 22;
   wsGlobalElNodeEqID(7,5,0) = 23;
   wsGlobalElNodeEqID(7,6,0) = 26;
   wsGlobalElNodeEqID(7,7,0) = 25;

   cell_map = Teuchos::rcp(new Tpetra_Map(numCells,0,comm));

   std::vector<Tpetra_GO> overlapped_nodes = {};

   for (int cell=0; cell<cell_map->getNodeNumElements(); ++cell)
      for(int node=0; node<nodes_per_element; ++node)
      if (std::find(overlapped_nodes.begin(),
                     overlapped_nodes.end(),
                     wsGlobalElNodeEqID(cell_map->getGlobalElement(cell),node,0))
            ==overlapped_nodes.end())
         overlapped_nodes.push_back(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell),node,0));

   overlapped_node_map = Teuchos::rcp(new Tpetra_Map(x_size, overlapped_nodes, 0, comm));

   Albany::WorksetConn wsLocalElNodeEqID("wsLocalElNodeEqID", cell_map->getNodeNumElements(), nodes_per_element, neq);

   for (int cell=0; cell<cell_map->getNodeNumElements(); ++cell)
      for(int node=0; node<nodes_per_element; ++node)
         wsLocalElNodeEqID(cell,node,0) = overlapped_node_map->getLocalElement(wsGlobalElNodeEqID(cell_map->getGlobalElement(cell),node,0));

   return wsLocalElNodeEqID;
}

}


#endif // EVAL_TESTSETUP
