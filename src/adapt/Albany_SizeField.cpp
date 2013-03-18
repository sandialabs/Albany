//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_SizeField.hpp"
#include "Albany_FMDBMeshStruct.hpp"
#include "Epetra_Import.h"

const double dist(double *p1, double *p2){

  return std::sqrt(p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]);

}

Albany::SizeField::SizeField(pMesh pm, Albany::FMDBDiscretization *disc_, const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol) :
	PWLsfield(pm),
        disc(disc_),
        solution(sol),
        ovlp_solution(ovlp_sol) {
}

Albany::SizeField::
~SizeField()
{
}

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
      FMDB_Vtx_GetCoord (node, xyz);

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
      FMDB_Vtx_GetCoord (node, xyz);

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

#if 0
int 
Albany::MeshAdapt::setSizeField(pMesh mesh, pSField pSizeField, void *vp){

  double L = 10.0;
  double R = 0.8;

  pMeshEnt node;
  double size, xyz[3];
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break;
    FMDB_Vtx_GetCoord (node, xyz);

    double circle= fabs(xyz[0] * xyz[0] +xyz[1] * xyz[1] + xyz[2] * xyz[2] - R*R);
    size = .1 * fabs(1. - exp (-circle*L)) + 1.e-3;
    pSizeField->setSize(node, size);
  }
  FMDB_PartEntIter_Del (node_it);
  return 1;
}
#endif











#if 0
// *********************************
//  isotropic mesh size specification - This code has not been run for long since it has not been updated since it's made.
// Just please get an idea of how to set  isotropic mesh size field from this example.
// *********************************

int main(int argc, char* argv[])
{
  pMeshMdl mesh;
  ...
  pSField field=new metricField(mesh);
  meshAdapt rdr(mesh,field,0,1);  // snapping off; do refinement only
  rdr.run(num_iteration,1, setSizeField);
  ..
}

/* size field for cube.msh */
int setSizeField(pMesh mesh, pSField pSizeField)
{
  double L = 10.0;
  double R = 0.8;

  pMeshEnt node;
  double size, xyz[3];
  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break;
    FMDB_Vtx_GetCoord (node, xyz);

    double circle= fabs(xyz[0] * xyz[0] +xyz[1] * xyz[1] + xyz[2] * xyz[2] - R*R);
    size = .1 * fabs(1. - exp (-circle*L)) + 1.e-3;
    pSizeField->setSize(node, size);
  }
  FMDB_PartEntIter_Del (node_it);
  return 1;
}

// *********************************
// anisotropic mesh size fields
// *********************************
int main(int argc, char* argv[])
{
  pMeshMdl mesh_instance;
  ...
  meshAdapt *rdr = new meshAdapt(mesh_instance, Analytical, 1);
  rdr->run(num_iteration,1, setSizeField);
  ...
}

int setSizeField(pMesh mesh, pSField field)
{
  double R0=1.; //.62;
  double L=3.;
  double center[]={1.0, 0.0, 0.0};
  double tol=0.01;
  double h[3], dirs[3][3], xyz[3], R, norm;
  R0=R0*R0;

  pVertex node;

  int iterEnd = FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, node_it);
  while (!iterEnd)
  {
    iterEnd = FMDB_PartEntIter_GetNext(node_it, node);
    if(iterEnd) break;
    FMDB_Vtx_GetCoord (node, xyz);
    R=dotProd(xyz,xyz);

    h[0] = .125 * fabs(1. - exp (-fabs(R-R0)*L)) + 0.00125;
    h[1] = .125;
    h[2] = .124;

    for(int i=0; i<3; i++)
      h[i] *= nSplit;

    norm=sqrt(R);
    if( norm>tol )
      {
        dirs[0][0]=xyz[0]/norm;
        dirs[0][1]=xyz[1]/norm;
        dirs[0][2]=xyz[2]/norm;
        if( xyz[0]*xyz[0] + xyz[1]*xyz[1] > tol*tol ) {
          dirs[1][0]=-1.0*xyz[1]/norm;
          dirs[1][1]=xyz[0]/norm;
          dirs[1][2]=0;
        } else {
          dirs[1][0]=-1.0*xyz[2]/norm;
          dirs[1][1]=0;
          dirs[1][2]=xyz[0]/norm;
        }
        crossProd(dirs[0],dirs[1],dirs[2]);
      }
    else
      {
        dirs[0][0]=1.0;
        dirs[0][1]=0.0;
        dirs[0][2]=0;
        dirs[1][0]=0.0;
        dirs[1][1]=1.0;
        dirs[1][2]=0;
        dirs[2][0]=0;
        dirs[1][2]=0;
        dirs[2][0]=0;
        dirs[2][1]=0;
        dirs[2][2]=1.0;
      }

    ((PWLsfield *)field)->setSize((pEntity)vt,dirs,h);
  }
  FMDB_PartEntIter_Del (node_it);

  double beta[]={2.5,2.5,2.5};
  ((PWLsfield *)field)->anisoSmooth(beta);
  return 1;
}
#endif
