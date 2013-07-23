//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ErrorSizeField.hpp"
#include "Albany_FMDBMeshStruct.hpp"
#include "Epetra_Import.h"
#include "PWLinearSField.h"

/********************************************************************/
Albany::ErrorSizeField::ErrorSizeField(Albany::FMDBDiscretization *disc_) :
        disc(disc_) {
/********************************************************************/

  mesh_struct = disc->getFMDBMeshStruct();
  mesh = mesh_struct->getMesh();

}

/********************************************************************/
Albany::ErrorSizeField::
~ErrorSizeField() {
/********************************************************************/

}

/********************************************************************/
void
Albany::ErrorSizeField::setParams(const Epetra_Vector *sol, const Epetra_Vector *ovlp_sol, double element_size) {
/********************************************************************/

  // -- set error parameter in some way --

  solution = sol;
  ovlp_solution = ovlp_sol;
  elem_size = element_size;

}
/********************************************************************/
void
Albany::ErrorSizeField::setError() {
/********************************************************************/
  
  // min and max global errors

  double min_err = std::numeric_limits<double>::max();
  double max_err = std::numeric_limits<double>::min();

  // find the "solution" tag - nodal displacement values
  
  disp_tag = mesh_struct->solution_field_tag;

  // ghost mesh regions with solution data
  
  FMDB_Tag_SetAutoMigrOn(mesh, disp_tag, FMDB_VERTEX);
  
#if 0
  FMDB_Mesh_CreateGhost(
			mesh,          /* mesh instance      */
  			FMDB_REGION,   /* ghost type         */
  			FMDB_VERTEX,   /* bridge type        */
  			1,             /* num. layers        */
  			1              /* create copies?     */
  			);
#endif

  // get mesh part
  
  pPart part;
  FMDB_Mesh_GetPart(mesh, 0, part);

  // initialize the error tag
  
  FMDB_Mesh_CreateTag(mesh, "error", SCUtil_DBL, 1, error_tag);
  initializeErrorTag(part);

  // compute the error for mesh regions on each part

  computeError(part);

  // set a new mesh size field for the elements

  FMDB_Mesh_CreateTag(mesh, "elem_h_new", SCUtil_DBL, 1, elem_h_new);
  computeElementalMeshSize(part);
  
  // set a new mesh size field for vertices based on the 
  // elemental mesh size field found above

  FMDB_Mesh_CreateTag(mesh, "vtx_h_new", SCUtil_DBL, 1, vtx_h_new);
  computeVertexMeshSize(part);

  // delete ghosts
  
  FMDB_Mesh_DelGhost(mesh); 
}

/********************************************************************/
int 
Albany::ErrorSizeField::computeSizeField(pPart part, pSField field) {
/********************************************************************/
  
  pMeshEnt vtx;
  double h[3], dirs[3][3], xyz[3];

  pPartEntIter vtx_iter;
  FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, vtx_iter);
  while (FMDB_PartEntIter_GetNext(vtx_iter, vtx)==SCUtil_SUCCESS)
  {
    
    double h_new;
    FMDB_Ent_GetDblTag(mesh, vtx, vtx_h_new, &h_new);
   
    h[0] = h_new;
    h[1] = h_new;
    h[2] = h_new;

    dirs[0][0]=1.0;
    dirs[0][1]=0.;
    dirs[0][2]=0.;
    dirs[1][0]=0.;
    dirs[1][1]=1.0;
    dirs[1][2]=0.;
    dirs[2][0]=0.;
    dirs[2][1]=0.;
    dirs[2][2]=1.0;

    ((PWLsfield *)field)->setSize(vtx,dirs,h);
  }
  FMDB_PartEntIter_Del(vtx_iter);

  //  double beta[]={1.75,1.75,1.75};
  //((PWLsfield *)field)->anisoSmooth(beta);

  return 1;
}

/********************************************************************/
void
Albany::ErrorSizeField::computeVertexMeshSize(pPart part) {
/********************************************************************/

  pMeshEnt vtx;
  pPartEntIter vtx_iter;
  
  FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, vtx_iter);
  while (FMDB_PartEntIter_GetNext(vtx_iter, vtx)==SCUtil_SUCCESS) {
    
    // get the new size for all adjacent regions to this vertex
    // and compute average size for this vertex
    
    std::vector<pMeshEnt> adj_regs;
    FMDB_Ent_GetAdj(vtx, FMDB_REGION, 1, adj_regs);
    
    double h_new = 0.0;
    
    for (int i=0; i<adj_regs.size(); i++) {
      double temp_h;
      FMDB_Ent_GetDblTag(mesh, adj_regs[i], elem_h_new, &temp_h);
      h_new += temp_h;
    }
      
    h_new /= adj_regs.size();

    FMDB_Ent_SetDblTag(mesh, vtx, vtx_h_new, h_new);

  }
  FMDB_PartEntIter_Del(vtx_iter);

}

/********************************************************************/
void
Albany::ErrorSizeField::computeElementalMeshSize(pPart part) {
/********************************************************************/

  // iterate over all regions in the mesh to calculate a new size field
  
  pMeshEnt tet;
  pPartEntIter tet_itr;
  
  FMDB_PartEntIter_Init(part, FMDB_REGION, FMDB_ALLTOPO, tet_itr);  
  while (SCUtil_SUCCESS == FMDB_PartEntIter_GetNext(tet_itr, tet)) {  
    
    // current tet size
    
    double h_current;
    h_current = computeTetSize(tet);

    // get the error defined on this tet
    
    double err_current;
    FMDB_Ent_GetDblTag(mesh, tet, error_tag, &err_current);
    
    // scaling factor for new element size
    
    double r;   
    r = (global_max_err - err_current) / (global_max_err - global_min_err);
    r = pow(r, 3);
    
    // min refinement is a factor of 0.05 of the current size
 
    if ( r < 0.05 )
      r = 0.05;
   
    // a new mesh size tagged to the tet 
    
    double h_new;
    h_new = r * h_current;
    
    FMDB_Ent_SetDblTag(mesh, tet, elem_h_new, h_new);
      
  }
  FMDB_PartEntIter_Del(tet_itr);

}

/********************************************************************/
void 
Albany::ErrorSizeField::initializeErrorTag(pPart part) {
/********************************************************************/
  
  // loop over all mesh regions and set this tag to be 0.0

  pMeshEnt tet;
  pPartEntIter tet_itr;

  FMDB_PartEntIter_Init(part, FMDB_REGION, FMDB_ALLTOPO, tet_itr);
  while (FMDB_PartEntIter_GetNext(tet_itr, tet)==SCUtil_SUCCESS) {
  
    FMDB_Ent_SetDblTag(mesh, tet, error_tag, 0.0);

  }
  FMDB_PartEntIter_Del(tet_itr);

}


/********************************************************************/
void
Albany::ErrorSizeField::computeError(pPart part) {
/********************************************************************/
  
  // min and max errors on this part
  
  double min_err = std::numeric_limits<double>::max();
  double max_err = std::numeric_limits<double>::min();

  // loop over mesh vertices

  pMeshEnt vtx;
  pPartEntIter vtx_itr;
  
  FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, vtx_itr);
  while (SCUtil_SUCCESS == FMDB_PartEntIter_GetNext(vtx_itr, vtx)) {  
    
    // get the regions adjacent to mesh entities

    std::vector<pMeshEnt> adj_regs;
    FMDB_Ent_GetAdj(vtx, FMDB_REGION, 1, adj_regs);

    // for the adjacent regions, compute strains and volumes
    
    MATH::vec volumes;
    MATH::matrix strains;
      
    for (int i=0; i<adj_regs.size(); i++) {
      
      double vol;
      vol = computeTetVolume( adj_regs[i] );
      volumes.push_back( vol );
      
      MATH::vec eps;
      eps = computeTetStrain( adj_regs[i], disp_tag );
      strains.push_back( eps );
      
    }

    // compute a constant volume weighted average "improved" strain
    // over the patch

    assert( volumes.size() == strains.size() );

    MATH::vec improved_strain;
    improved_strain.resize(6);
    for (int i=0; i<improved_strain.size(); i++) {
      improved_strain[i] = 0.0;
    }
    
    double tot_vol = 0.0;
    
    for (int i=0; i<volumes.size(); i++) {    
      tot_vol += volumes[i];
      for (int j=0; j<6; j++) {
	improved_strain[j] += volumes[i]*strains[i][j];
      } 
    }
    
    for (int i=0; i<6; i++) {
      improved_strain[i] = improved_strain[i] / tot_vol;
    }

    // define the e in an element as improved strain - element strain
    
    for (int i=0; i<adj_regs.size(); i++) {
      
      MATH::vec e;
      e.resize(6);

      for (int j=0; j<6; j++) {
	e[j] = improved_strain[j] - strains[i][j];	
      }

      // call the L2 norm of e the "error"

      double err;
      err = computeL2Norm(e);
      
      if (err < min_err) 
	min_err = err;

      if (err > max_err)
	max_err = err;
      
      // tag this error to the current element if it is larger than
      // previous errors
      
      double old_err;
      FMDB_Ent_GetDblTag(mesh, adj_regs[i], error_tag, &old_err);

      if ( err > old_err ) {	
	FMDB_Ent_SetDblTag(mesh, adj_regs[i], error_tag, err);    
      }
    }
  }
  FMDB_PartEntIter_Del(vtx_itr);
  
  // find global minimum and maximum errors

  MPI_Allreduce(&max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&min_err, &global_min_err, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  
}

/********************************************************************/
double 
Albany::ErrorSizeField::computeTetSize(pMeshEnt tet) {
/********************************************************************/

  std::vector<pMeshEnt> adj_edges;
  FMDB_Ent_GetAdj(tet, FMDB_EDGE, 1, adj_edges);
  
  double avg_size = 0.0;
  int num_edges = 0;

  for (int i=0; i<adj_edges.size(); i++) {
    num_edges++;
    double len = sqrt(E_lengthSq(adj_edges[i]));
    avg_size += len;
  }
 
  avg_size /= num_edges;

  return avg_size;

}

/********************************************************************/
double
Albany::ErrorSizeField::computeTetVolume(pMeshEnt tet) {
/********************************************************************/
  
  // region must be a tet
  
  int topo;
  FMDB_Ent_GetTopo(tet, &topo);
  assert( topo = FMDB_TET );
  
  // get vertices that define tet

  std::vector<pMeshEnt> vtx;
  FMDB_Ent_GetAdj(tet, FMDB_VERTEX, 1, vtx);
  
  // get xyz coordinates of vertices

  double c0[3], c1[3], c2[3], c3[3];
  
  FMDB_Vtx_GetCoord(vtx[0], c0);
  FMDB_Vtx_GetCoord(vtx[1], c1);
  FMDB_Vtx_GetCoord(vtx[2], c2);
  FMDB_Vtx_GetCoord(vtx[3], c3);
  
  // compute edge vectors 
  // based on the 3 edges adjacent to vtx[0]

  double e0[3], e1[3], e2[3];

  for (int i=0; i<3; i++) {
    e0[i] = c1[i] - c0[i];
    e1[i] = c2[i] - c0[i];
    e2[i] = c3[i] - c0[i];    
  } 

  // compute volume

  double vol;
  
  vol = (1.0/6.0)*( e0[0]*( e1[1]*e2[2] - e1[2]*e2[1] ) -
		    e0[1]*( e1[0]*e2[2] - e1[2]*e2[0] ) +
		    e0[2]*( e1[0]*e2[1] - e1[1]*e2[0] ) );
  
  return vol;

}

/********************************************************************/
MATH::vec
Albany::ErrorSizeField::computeTetStrain(pMeshEnt tet, pTag disp_tag) {
/********************************************************************/
  
  // region must be a tet
  
  int topo;
  FMDB_Ent_GetTopo(tet, &topo);
  assert( topo = FMDB_TET );
  
  // get vertices that define tet

  std::vector<pMeshEnt> vtx;
  FMDB_Ent_GetAdj(tet, FMDB_VERTEX, 1, vtx);
  
  // get xyz coordinates of vertices

  double c0[3], c1[3], c2[3], c3[3];
  
  FMDB_Vtx_GetCoord(vtx[0], c0);
  FMDB_Vtx_GetCoord(vtx[1], c1);
  FMDB_Vtx_GetCoord(vtx[2], c2);
  FMDB_Vtx_GetCoord(vtx[3], c3);
  
  // get displacements associated with vertices
  
  int disp_size;

  double* d0 = new double[3];
  double* d1 = new double[3];
  double* d2 = new double[3];
  double* d3 = new double[3];
  
  FMDB_Ent_GetDblArrTag(mesh, vtx[0], disp_tag, &d0, &disp_size);
  FMDB_Ent_GetDblArrTag(mesh, vtx[1], disp_tag, &d1, &disp_size);
  FMDB_Ent_GetDblArrTag(mesh, vtx[2], disp_tag, &d2, &disp_size);
  FMDB_Ent_GetDblArrTag(mesh, vtx[3], disp_tag, &d3, &disp_size);
  
  // entries for the Jacobian
  // ONLY VALID FOR LINEAR TET W/ LAGRANGE SHAPE FUNC.
  // N1 = 1- xi - eta - zeta  
  // N2 = xi
  // N3 = eta
  // N4 = zeta

  MATH::matrix J;
  J.resize(3);
  for (int i=0; i<J.size(); i++) {
    J[i].resize(3);
  }

  J[0][0] = c1[0] - c0[0];   J[0][1] = c1[1] - c0[1];   J[0][2] = c1[2] - c0[2];
  J[1][0] = c2[0] - c0[0];   J[1][1] = c2[1] - c0[1];   J[1][2] = c2[2] - c0[2];
  J[2][0] = c3[0] - c0[0];   J[2][1] = c3[1] - c0[1];   J[2][2] = c3[2] - c0[2];

  // find inverse of Jacobian
  
  MATH::matrix invJ;
  invJ = computeInverse3x3(J);

  // derivatives of shape functions wrt local coordinates (xi, eta, zeta)
  
  MATH::vec dN1dXi;
  dN1dXi.resize(3);
  dN1dXi[0] = -1.0;
  dN1dXi[1] = -1.0;
  dN1dXi[2] = -1.0;
  
  MATH::vec dN2dXi;
  dN2dXi.resize(3);
  dN2dXi[0] = 1.0;
  dN2dXi[1] = 0.0;
  dN2dXi[2] = 0.0;

  MATH::vec dN3dXi;
  dN3dXi.resize(3);
  dN3dXi[0] = 0.0;
  dN3dXi[1] = 1.0;
  dN3dXi[2] = 0.0;

  MATH::vec dN4dXi;
  dN4dXi.resize(3);
  dN4dXi[0] = 0.0;
  dN4dXi[1] = 0.0;
  dN4dXi[2] = 1.0;

  // compute derivatives of shape functions wrt global coordinates (x,y,z)

  MATH::vec dN1dx;
  dN1dx.resize(3);
  dN1dx = multiplyMatrixVec(invJ, dN1dXi);

  MATH::vec dN2dx;
  dN2dx.resize(3);
  dN2dx = multiplyMatrixVec(invJ, dN2dXi);

  MATH::vec dN3dx;
  dN3dx.resize(3);
  dN3dx = multiplyMatrixVec(invJ, dN3dXi);

  MATH::vec dN4dx;
  dN4dx.resize(3);
  dN4dx = multiplyMatrixVec(invJ, dN4dXi);

  // deformation matrix B
  
  MATH::matrix B;
  B.resize(6);
  for (int i=0; i<B.size(); i++) {
    B[i].resize(12);
  }
  
  // B - first row
  
  B[0][0] = dN1dx[0]; B[0][1] = 0.0;  B[0][2] = 0.0;
  B[0][3] = dN2dx[0]; B[0][4] = 0.0;  B[0][5] = 0.0;
  B[0][6] = dN3dx[0]; B[0][7] = 0.0;  B[0][8] = 0.0;
  B[0][9] = dN4dx[0]; B[0][10] = 0.0; B[0][11] = 0.0;

  // B - second row
  
  B[1][0] = 0.0; B[1][1] = dN1dx[1];  B[1][2] = 0.0;
  B[1][3] = 0.0; B[1][4] = dN2dx[1];  B[1][5] = 0.0;
  B[1][6] = 0.0; B[1][7] = dN3dx[1];  B[1][8] = 0.0;
  B[1][9] = 0.0; B[1][10] = dN4dx[1]; B[1][11] = 0.0;
  
  // B - third row
  
  B[2][0] = 0.0; B[2][1] = 0.0;  B[2][2] = dN1dx[2];
  B[2][3] = 0.0; B[2][4] = 0.0;  B[2][5] = dN2dx[2];
  B[2][6] = 0.0; B[2][7] = 0.0;  B[2][8] = dN3dx[2];
  B[2][9] = 0.0; B[2][10] = 0.0; B[2][11] = dN4dx[2];

  // B - fourth row

  B[3][0] = dN1dx[1]; B[3][1] = dN1dx[0];  B[3][2] = 0.0;
  B[3][3] = dN2dx[1]; B[3][4] = dN2dx[0];  B[3][5] = 0.0;
  B[3][6] = dN3dx[1]; B[3][7] = dN3dx[0];  B[3][8] = 0.0;
  B[3][9] = dN4dx[1]; B[3][10] = dN4dx[0]; B[3][11] = 0.0;

  // B - fifth row

  B[4][0] = 0.0; B[4][1] = dN1dx[2];  B[4][2] = dN1dx[1];
  B[4][3] = 0.0; B[4][4] = dN2dx[2];  B[4][5] = dN2dx[1];
  B[4][6] = 0.0; B[4][7] = dN3dx[2];  B[4][8] = dN3dx[1];
  B[4][9] = 0.0; B[4][10] = dN4dx[2]; B[4][11] = dN4dx[1];

  // B - sixth row

  B[5][0] = dN1dx[2]; B[5][1] = 0.0;  B[5][2] = dN1dx[0];
  B[5][3] = dN2dx[2]; B[5][4] = 0.0;  B[5][5] = dN2dx[0];
  B[5][6] = dN3dx[2]; B[5][7] = 0.0;  B[5][8] = dN3dx[0];
  B[5][9] = dN4dx[2]; B[5][10] = 0.0; B[5][11] = dN4dx[0];
  
  // displacement vector

  MATH::vec u;
  
  for (int i=0; i<3; i++) {
    u.push_back(d0[i]);
  }
  for (int i=0; i<3; i++) {
    u.push_back(d1[i]);
  }
  for (int i=0; i<3; i++) {
    u.push_back(d2[i]);
  }
  for (int i=0; i<3; i++) {
    u.push_back(d3[i]);
  }

  // compute strain 
  
  MATH::vec eps;
  eps = multiplyMatrixVec(B, u);

  // memory clean-up

  delete [] d0;
  delete [] d1;
  delete [] d2;
  delete [] d3;

  return eps;
  
}

/********************************************************************/
double
Albany::ErrorSizeField::computeL2Norm(MATH::vec x) {
/********************************************************************/

  double norm = 0.0;

  for (int i=0; i<x.size(); i++) {
    norm += x[i]*x[i];
  }

  norm = sqrt(norm);

  return norm;
}

/********************************************************************/
MATH::vec
Albany::ErrorSizeField::multiplyMatrixVec(MATH::matrix A, MATH::vec b) {
/********************************************************************/
  
  // check dimensional validity

  int m = A[0].size();
  assert ( m == b.size() );
  
  // resultant vector
  
  MATH::vec x;
  x.resize(A.size());
  
  double sum;
  for (int i=0; i<A.size(); i++) {
    sum = 0.0;
    for (int j=0; j<m; j++ ) {
      sum += A[i][j]*b[j];
    }
    x[i] = sum;
  }
 
  return x;  

}

/********************************************************************/
MATH::matrix
Albany::ErrorSizeField::computeInverse3x3(MATH::matrix A) {
/********************************************************************/

  // check A is actually 3x3

  assert ( A.size() == 3 );

  for (int i=0; i<A.size(); i++ ) {
    assert ( A[i].size() == 3 );
  }

  // determinant of matrix
  
  double detA;
  
  detA = ( A[0][0] * ( A[1][1]*A[2][2] - A[1][2]*A[2][1] ) -
	   A[0][1] * ( A[1][0]*A[2][2] - A[1][2]*A[2][0] ) +
	   A[0][2] * ( A[1][0]*A[2][1] - A[1][1]*A[2][0] ) );
  
  // allocate some space for inverse of A

  MATH::matrix invA;
  
  invA.resize(3);
  
  for (int i=0; i<invA.size(); i++) {
    invA[i].resize(3);
  }

  // cofactor calculations
  
  invA[0][0] = A[1][1]*A[2][2] - A[2][1]*A[1][2];
  invA[0][1] = A[0][2]*A[2][1] - A[2][2]*A[0][1];
  invA[0][2] = A[0][1]*A[1][2] - A[1][1]*A[0][2];
  
  invA[1][0] = A[1][2]*A[2][0] - A[2][2]*A[1][0];
  invA[1][1] = A[0][0]*A[2][2] - A[2][0]*A[0][2];
  invA[1][2] = A[0][2]*A[1][0] - A[1][2]*A[0][0];
  
  invA[2][0] = A[1][0]*A[2][1] - A[2][0]*A[1][1];
  invA[2][1] = A[0][1]*A[2][0] - A[2][1]*A[0][0];
  invA[2][2] = A[0][0]*A[1][1] - A[1][0]*A[0][1];
  
  // divide by detA to find inverse A
  
  for (int i=0; i<invA.size(); i++) {
    for (int j=0; j<invA[i].size(); j++) {
      invA[i][j] = invA[i][j] / detA;
    }
  }

  return invA;
}

