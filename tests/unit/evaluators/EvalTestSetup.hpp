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

}


#endif // EVAL_TESTSETUP
