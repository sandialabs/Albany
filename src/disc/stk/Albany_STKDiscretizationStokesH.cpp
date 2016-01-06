//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <limits>

#include "Albany_Utils.hpp"
#include "Albany_STKDiscretizationStokesH.hpp"
#include "Albany_NodalGraphUtils.hpp"
#include "Albany_STKNodeFieldContainer.hpp"
#include "Albany_BucketArray.hpp"

#include <string>
#include <iostream>
#include <fstream>

#include <Shards_BasicTopologies.hpp>

#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_Basis.hpp>
#include <Intrepid2_HGRAD_QUAD_Cn_FEM.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <PHAL_Dimension.hpp>

#include <stk_mesh/base/FEMHelpers.hpp>

Albany::STKDiscretizationStokesH::
STKDiscretizationStokesH(Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct_,
                  const Teuchos::RCP<const Teuchos_Comm>& commT_,
                  const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_) :
   Albany::STKDiscretization(stkMeshStruct_,  commT_, rigidBodyModes_)
{}

Albany::STKDiscretizationStokesH::~STKDiscretizationStokesH() {}


void Albany::STKDiscretizationStokesH::computeGraphs()
{
  std::map<int, stk::mesh::Part*>::iterator pv = stkMeshStruct->partVec.begin();
  int nodes_per_element =  metaData.get_cell_topology(*(pv->second)).getNodeCount();
// int nodes_per_element_est =  metaData.get_cell_topology(*(stkMeshStruct->partVec[0])).getNodeCount();

  // Loads member data:  overlap_graph, numOverlapodes, overlap_node_map, coordinates, graphs

  const unsigned int n3dEq = 2;

  overlap_graphT = Teuchos::null; // delete existing graph happens here on remesh

  overlap_graphT = Teuchos::rcp(new Tpetra_CrsGraph(overlap_mapT, neq*nodes_per_element));

  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  stk::mesh::get_selected_entities( select_owned_in_part ,
            bulkData.buckets( stk::topology::ELEMENT_RANK ) ,
            cells );

  if (commT->getRank()==0)
    *out << "STKDisc: " << cells.size() << " elements on Proc 0 " << std::endl;

  GO row, col;
  Teuchos::ArrayView<GO> colAV;

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *getLayeredMeshNumbering();
  int numLayers = layeredMeshNumbering.numLayers;

  for (std::size_t i=0; i < cells.size(); i++) {
    stk::mesh::Entity e = cells[i];
    stk::mesh::Entity const* node_rels = bulkData.begin_nodes(e);
    const size_t num_nodes = bulkData.num_nodes(e);

    // loop over local nodes
    for (std::size_t j=0; j < num_nodes; j++) {
      stk::mesh::Entity rowNode = node_rels[j];

      // loop over eqs
      for (std::size_t k=0; k < n3dEq; k++) {
        row = this->getGlobalDOF(this->gid(rowNode), k);
        for (std::size_t l=0; l < num_nodes; l++) {
          stk::mesh::Entity colNode = node_rels[l];
          for (std::size_t m=0; m < n3dEq; m++) {
            col = this->getGlobalDOF(this->gid(colNode), m);
            colAV = Teuchos::arrayView(&col, 1);
            overlap_graphT->insertGlobalIndices(row, colAV);
          }
        }
      }

      if(neq > n3dEq)
      {
        row = this->getGlobalDOF(this->gid(rowNode), n3dEq);
        GO node_gid = this->gid(rowNode);
        LO base_id, ilayer;
        int node_lid = overlap_node_mapT->getLocalElement(node_gid);
        layeredMeshNumbering.getIndices(node_lid, base_id, ilayer);
        if(ilayer == numLayers) {
          for (std::size_t l=0; l < num_nodes; l++) {
            stk::mesh::Entity colNode = node_rels[l];
            node_gid = this->gid(colNode);
            node_lid = overlap_node_mapT->getLocalElement(node_gid);
            layeredMeshNumbering.getIndices(node_lid, base_id, ilayer);
            if(ilayer == numLayers) {
              for (unsigned int il_col=0; il_col<numLayers+1; il_col++) {
                LO inode = layeredMeshNumbering.getId(base_id, il_col);
                GO gnode = overlap_node_mapT->getGlobalElement(inode);
                for (std::size_t m=0; m < n3dEq; m++) {
                  col = getGlobalDOF(gnode, m);
                  overlap_graphT->insertGlobalIndices(row, Teuchos::arrayView(&col, 1));
                  overlap_graphT->insertGlobalIndices(col, Teuchos::arrayView(&row, 1));
                }
                col = getGlobalDOF(gnode, n3dEq);
                overlap_graphT->insertGlobalIndices(row, Teuchos::arrayView(&col, 1));
              }
            }
          }
        }
        else
          overlap_graphT->insertGlobalIndices(row, Teuchos::arrayView(&row, 1)); //insert diagonal elements
      }
    }
  }
  overlap_graphT->fillComplete();

  // Create Owned graph by exporting overlap with known row map
  graphT = Teuchos::null; // delete existing graph happens here on remesh

  graphT = Teuchos::rcp(new Tpetra_CrsGraph(mapT, nonzeroesPerRow(neq)));

  // Create non-overlapped matrix using two maps and export object
  Teuchos::RCP<Tpetra_Export> exporterT = Teuchos::rcp(new Tpetra_Export(overlap_mapT, mapT));
  graphT->doExport(*overlap_graphT, *exporterT, Tpetra::INSERT);
  graphT->fillComplete();
}
