//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_UnifRefSizeField.hpp"
#include "Albany_FMDBMeshStruct.hpp"
#include "Epetra_Import.h"
#include "PWLinearSField.h"



const double dist(double *p1, double *p2){

  return std::sqrt(p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]);

}

Albany::UnifRefSizeField::UnifRefSizeField(Albany::FMDBDiscretization *disc_) :
        disc(disc_)
{
}

Albany::UnifRefSizeField::
~UnifRefSizeField()
{
}

void 
Albany::UnifRefSizeField::setParams(const Epetra_Vector *sol, const Epetra_Vector *ovlp_sol, double element_size){

  solution = sol;
  ovlp_solution = ovlp_sol;
  elem_size = element_size;

}

//int Albany::UnifRefSizeField::computeSizeField(pPart part, pSField field, void *vp){
int Albany::UnifRefSizeField::computeSizeField(pPart part, pSField field){

   pVertex vt;
   double h[3], dirs[3][3];
   
   static int numCalls = 0;
   static double initAvgEdgeLen = 0;
   static double currGlobMin = 0, currGlobMax = 0, currGlobAvg = 0;
   if (0 == numCalls) {
      getCurrentSize(part, currGlobMin, currGlobMax, initAvgEdgeLen);
   } else {
      getCurrentSize(part, currGlobMin, currGlobMax, currGlobAvg);
   }
   
  double minSize = std::numeric_limits<double>::max();
  double maxSize = std::numeric_limits<double>::min();
        
   VIter vit = M_vertexIter(part);
   while (vt = VIter_next(vit)) {
      const double sz = 0.5 * initAvgEdgeLen;
      if ( sz < minSize ) minSize = sz;
      if ( sz > maxSize ) maxSize = sz;   
      for (int i = 0; i < 3; i++) {
         h[i] = sz;
      }

      dirs[0][0] = 1.0;
      dirs[0][1] = 0.;
      dirs[0][2] = 0.;
      dirs[1][0] = 0.;
      dirs[1][1] = 1.0;
      dirs[1][2] = 0.;
      dirs[2][0] = 0.;
      dirs[2][1] = 0.;
      dirs[2][2] = 1.0;

      ((PWLsfield *) field)->setSize((pEntity) vt, dirs, h);
   }

   VIter_delete(vit);
//   double beta[] = {1.5, 1.5, 1.5};
//   ((PWLsfield *) field)->anisoSmooth(beta);
   

   double globMin = 0;
   double globMax = 0;
   MPI_Reduce(&minSize, &globMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&maxSize, &globMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   if (0 == SCUTIL_CommRank()) {
      if ( 0 == numCalls ) {
         printf("%s initial edgeLength avg %f min %f max %f\n", __FUNCTION__, initAvgEdgeLen, currGlobMin, currGlobMax);
         printf("%s target edgeLength min %f max %f\n", __FUNCTION__, globMin, globMax);
      } else {
         printf("%s current edgeLength avg %f min %f max %f\n", __FUNCTION__, currGlobAvg, currGlobMin, currGlobMax);      
      }
      fflush(stdout);      
   }   
   
   numCalls++;
   
   return 1;
}

int 
Albany::UnifRefSizeField::getCurrentSize(pPart part, double& globMinSize, double& globMaxSize, double& globAvgSize) {

   EIter eit = M_edgeIter(part);
   pEdge edge;
   int numEdges = 0;
   double avgSize = 0.;
   double minSize = std::numeric_limits<double>::max();
   double maxSize = std::numeric_limits<double>::min();
   
   while (edge = EIter_next(eit)) {
      numEdges++;
      double len = sqrt(E_lengthSq(edge));
      avgSize += len;
      if ( len < minSize ) minSize = len;
      if ( len > maxSize ) maxSize = len;
   }
   EIter_delete(eit);
   avgSize /= numEdges; 

   MPI_Allreduce(&avgSize, &globAvgSize, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&maxSize, &globMaxSize, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(&minSize, &globMinSize, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   globAvgSize /= SCUTIL_CommSize();

}


#if 0
int Albany::SizeField::computeSizeField(){

  // Loop over the elements in the mesh
  pPartEntIter cell_it;
  pMeshEnt cell;
  std::vector<pMeshEnt> rel;
  double *xyz = new double[3];
  double *center = new double[3];

  const Teuchos::RCP<const Epetra_Map> overlap_map = disc->getOverlapMap();
  const Teuchos::RCP<const Epetra_Map> overlap_node_map = disc->getOverlapNodeMap();
  const Teuchos::RCP<const Epetra_Map> node_map = disc->getNodeMap();
  const Teuchos::RCP<Albany::FMDBMeshStruct> fmdbMeshStruct = disc->getFMDBMeshStruct();

  // Build an Epetra_Vector to hold the distances from each node to the element center
  Epetra_Vector cent_dist(*node_map);
  Epetra_Vector dist_sum(*node_map);

  // Build a length field from the nodes of each cell to the cell center
  int iterEnd = FMDB_PartEntIter_Init(backMesh, FMDB_REGION, FMDB_ALLTOPO, cell_it);
  while (!iterEnd){

    iterEnd = FMDB_PartEntIter_GetNext(cell_it, cell);
    if(iterEnd) break; 

    // Get the nodes belonging to the cell
    rel.clear();
    FMDB_Ent_GetAdj(cell, FMDB_VERTEX, 1, rel);

    // loop over local nodes, calculate the element center
    center[0] = center[1] = center[2] = 0.0;

    for (std::size_t j=0; j < rel.size(); j++){

      pMeshEnt node = rel[j];
      FMDB_Vtx_GetCoord (node, &xyz);

      // Displace the nodes
/*
      for (std::size_t j=0; j < fmdbMeshStruct->neq; j++){
         int local_id = overlap_map->LID(disc->getOverlapDOF(FMDB_Ent_ID(node),j));
         center[j] += xyz[j] + ovlp_solution[local_id];
      }
*/

      // Do not displace the nodes
      center[0] += xyz[0];
      center[1] += xyz[1];
      center[2] += xyz[2];

    }

    center[0] /= (double)rel.size();
    center[1] /= (double)rel.size();
    center[2] /= (double)rel.size();

    // Accumulate the distance from the node to the cell center
    for (std::size_t j=0; j < rel.size(); j++){

      pMeshEnt node = rel[j];
      FMDB_Vtx_GetCoord (node, &xyz);

      int local_node = node_map->LID(FMDB_Ent_ID(node));
      cent_dist[local_node] += distance(center, xyz);
      dist_sum[local_node] += 1.0;
    }

  }
  FMDB_PartEntIter_Del (cell_it);

  // Import off processor centroid sums
  Epetra_Import importer(*overlap_node_map, *node_map);
  Epetra_Vector ovlp_cent_dist(*overlap_node_map);
  Epetra_Vector ovlp_dist_sum(*overlap_node_map);

  ovlp_cent_dist.Import(cent_dist, importer, Add);
  ovlp_dist_sum.Import(dist_sum, importer, Add);

  const std::vector<pMeshEnt>& owned_nodes = disc->getOwnedNodes();

  for(std::size_t node = 0; node < owned_nodes.size(); node++){

    pMeshEnt l_node = owned_nodes[node];
    int index = overlap_node_map->LID(node_map->GID(node));
    double size = ovlp_cent_dist[index] / ovlp_dist_sum[index];
    this->setSize(l_node, size);

  }

  delete [] xyz;
  delete [] center;

  return 1;

}
#endif


