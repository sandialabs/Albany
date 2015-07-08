//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Piro_StratimikosUtils.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Albany_NullSpaceUtils.hpp"

namespace Albany {
namespace {
// Copied from Trilinos/packages/ml/src/Utils/ml_rbm.c.
void Coord2RBM(
  const LO Nnodes,
  double const* const x, double const* const y, double const* const z,
  const int Ndof, const int NscalarDof, const int NSdim,
  double* const rbm)
{
  LO vec_leng, offset;
  int ii, jj, dof;

  vec_leng = Nnodes*Ndof;
  for (LO i = 0; i < Nnodes*Ndof*(NSdim + NscalarDof); i++)
    rbm[i] = 0.0;

  for (LO node = 0 ; node < Nnodes; node++) {
    dof = node*Ndof;
    switch( Ndof - NscalarDof ) {
    case 6:
      for (ii=3;ii<6+NscalarDof;ii++) { /* lower half = [ 0 I ] */
        for (jj=0;jj<6+NscalarDof;jj++) {
          offset = dof+ii+jj*vec_leng;
          rbm[offset] = (ii==jj) ? 1.0 : 0.0;
        }
      }
      /* There is no break here and that is on purpose */
    case 3:
      for (ii=0;ii<3+NscalarDof;ii++) { /* upper left = [ I ] */
        for (jj=0;jj<3+NscalarDof;jj++) {
          offset = dof+ii+jj*vec_leng;
          rbm[offset] = (ii==jj) ? 1.0 : 0.0;
        }
      }
      for (ii=0;ii<3;ii++) { /* upper right = [ Q ] */
        for (jj=3+NscalarDof;jj<6+NscalarDof;jj++) {
          offset = dof+ii+jj*vec_leng;
          // std::cout <<"jj " << jj << " " << ii + jj << std::endl;
          if(ii == jj-3-NscalarDof) rbm[offset] = 0.0;
          else {
            if (ii+jj == 4+NscalarDof) rbm[offset] = z[node];
            else if ( ii+jj == 5+NscalarDof ) rbm[offset] = y[node];
            else if ( ii+jj == 6+NscalarDof ) rbm[offset] = x[node];
            else rbm[offset] = 0.0;
          }
        }
      }
      ii = 0; jj = 5+NscalarDof; offset = dof+ii+jj*vec_leng; rbm[offset] *= -1.0;
      ii = 1; jj = 3+NscalarDof; offset = dof+ii+jj*vec_leng; rbm[offset] *= -1.0;
      ii = 2; jj = 4+NscalarDof; offset = dof+ii+jj*vec_leng; rbm[offset] *= -1.0;
      break;

    case 2:
      for (ii=0;ii<2+NscalarDof;ii++) { /* upper left = [ I ] */
        for (jj=0;jj<2+NscalarDof;jj++) {
          offset = dof+ii+jj*vec_leng;
          rbm[offset] = (ii==jj) ? 1.0 : 0.0;
        }
      }
      for (ii=0;ii<2+NscalarDof;ii++) { /* upper right = [ Q ] */
        for (jj=2+NscalarDof;jj<3+NscalarDof;jj++) {
          offset = dof+ii+jj*vec_leng;
          if (ii == 0) rbm[offset] = -y[node];
          else {
            if (ii == 1) { rbm[offset] =  x[node];}
            else rbm[offset] = 0.0;
          }
        }
      }
      break;

    case 1:
      for (ii = 0; ii<1+NscalarDof; ii++) {
        for (jj=0; jj<1+NscalarDof; jj++) {
          offset = dof+ii+jj*vec_leng;
          rbm[offset] = (ii == jj) ? 1.0 : 0.0;
        }
      }
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Coord2RBM: Ndof = " << Ndof << " not implemented\n");
    } /*switch*/

  } /*for (node = 0 ; node < Nnodes; node++)*/

  return;
} /*Coord2RBM*/

//IKT, 6/28/15: the following set RBMs for non-elasticity problems.
void Coord2RBM_nonElasticity(
  const LO Nnodes,
  double const* const x, double const* const y, double const* const z,
  const int Ndof, const int NscalarDof, const int NSdim,
  double* const rbm)
{
  std::cout << "setting RBMs in Coord2RBM_nonElasticity!" << std::endl; 
  LO vec_leng, offset;
  int ii, jj, dof;

  vec_leng = Nnodes*Ndof;
  for (LO i = 0; i < Nnodes*Ndof*(NSdim + NscalarDof); i++)
    rbm[i] = 0.0;

  std::cout << "...case: " << Ndof - NscalarDof << std::endl; 
  for (LO node = 0 ; node < Nnodes; node++) {
    dof = node*Ndof;
    switch( Ndof - NscalarDof ) {
    case 3:
      for (ii=0;ii<2;ii++) { /* upper right = [ Q ] -- xy rotation only */
        jj = 2+NscalarDof; 
        offset = dof+ii+jj*vec_leng;
        // std::cout <<"jj " << jj << " " << ii + jj << std::endl;
        if (ii == 0) 
          rbm[offset] = y[node]; 
        else if (ii == 1)
          rbm[offset] = x[node]; 
      }
      ii = 0; jj = 2+NscalarDof; offset = dof+ii+jj*vec_leng; rbm[offset] *= -1.0;
      /* There is no break here and that is on purpose */
    case 2:
      for (ii=0;ii<2+NscalarDof;ii++) { /* upper left = [ I ] */
        for (jj=0;jj<2+NscalarDof;jj++) {
          offset = dof+ii+jj*vec_leng;
          rbm[offset] = (ii==jj) ? 1.0 : 0.0;
        }
      }
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Coord2RBM_nonElasticity: Ndof = " << Ndof << " not implemented\n");
    } /*switch*/

  } /*for (node = 0 ; node < Nnodes; node++)*/

  return;
} /*Coord2RBM_nonElasticity*/

void subtractCentroid(
  const Teuchos::RCP<const Tpetra_Map>& node_map, const int ndim,
  std::vector<ST>& v)
{
  const int nnodes = v.size() / ndim;

  ST centroid[3]; // enough for up to 3d
  {
    ST sum[3];
    ST* pv = &v[0];
    for (int i = 0; i < ndim; ++i) {
      sum[i] = 0;
      for (int j = 0; j < nnodes; ++j) sum[i] += pv[j];
      pv += nnodes;
    }
    Teuchos::reduceAll<int, ST>(*node_map->getComm(), Teuchos::REDUCE_SUM, ndim,
                                sum, centroid);
    const GO ng = node_map->getGlobalNumElements();
    for (int i = 0; i < ndim; ++i) centroid[i] /= ng;
  }

  ST* pv = &v[0];
  for (int i = 0; i < ndim; ++i) {
    for (Teuchos::ArrayRCP<ST>::size_type j = 0; j < nnodes; ++j)
      pv[j] -= centroid[i];
    pv += nnodes;
  }
}
} // namespace

RigidBodyModes::RigidBodyModes(int numPDEs_)
  : numPDEs(numPDEs_), numElasticityDim(0), nullSpaceDim(0), numSpaceDim(0),
    numScalar(0), mlUsed(false), mueLuUsed(false), setNonElastRBM(false)
{}

void RigidBodyModes::
setPiroPL(const Teuchos::RCP<Teuchos::ParameterList>& piroParams)
{
  const Teuchos::RCP<Teuchos::ParameterList>
    stratList = Piro::extractStratimikosParams(piroParams);

  mlUsed = mueLuUsed = false;
  if (Teuchos::nonnull(stratList) &&
      stratList->isParameter("Preconditioner Type")) {
    const std::string&
      ptype = stratList->get<std::string>("Preconditioner Type");
    if (ptype == "ML") {
      plist = sublist(sublist(sublist(stratList, "Preconditioner Types"),
                              ptype), "ML Settings");
      mlUsed = true;
    }
    else if (ptype == "MueLu" || ptype == "MueLu-Tpetra") {
      plist = sublist(sublist(stratList, "Preconditioner Types"), ptype);
      mueLuUsed = true;
    }
  }
}

void RigidBodyModes::
updatePL(const Teuchos::RCP<Teuchos::ParameterList>& mlParams)
{
  plist = mlParams;
}

void RigidBodyModes::
resize(const int numSpaceDim_, const LO numNodes)
{
  numSpaceDim = numSpaceDim_;
  xyz.resize(numSpaceDim * (numNodes == 0 ? 1 : numNodes));
  if(nullSpaceDim > 0)
    if (setNonElastRBM == true)
      rr.resize((nullSpaceDim + numScalar) * numSpaceDim * numNodes);
    else
      rr.resize((nullSpaceDim + numScalar) * numPDEs * numNodes);
}

void RigidBodyModes::
getCoordArrays(double*& xx, double*& yy, double*& zz)
{
  const LO nn = xyz.size() / numSpaceDim;
  xx = &xyz[0];
  yy = zz = NULL;
  if (numSpaceDim > 1) {
    yy = &xyz[0] + nn;
    if (numSpaceDim > 2)
      zz = &xyz[0] + 2*nn;
  }
}
  
double* RigidBodyModes::getCoordArray() { return &xyz[0]; }

void RigidBodyModes::setParameters(
  const int numPDEs_, const int numElasticityDim_, const int numScalar_,
  const int nullSpaceDim_, const bool setNonElastRBM_)
{
  numPDEs = numPDEs_;
  numElasticityDim = numElasticityDim_;
  numScalar = numScalar_;
  nullSpaceDim = nullSpaceDim_;
  setNonElastRBM = setNonElastRBM_; 
}

void RigidBodyModes::
setCoordinates(const Teuchos::RCP<const Tpetra_Map>& node_map)
{
  const LO numNodes = xyz.size() / numSpaceDim;
  TEUCHOS_TEST_FOR_EXCEPTION(
    node_map->getNodeNumElements() != numNodes,
    std::logic_error,
    "The non-overlap node map passed to informMueLu should have as many"
    " elements as there are owned nodes.");
  TEUCHOS_TEST_FOR_EXCEPTION(
    !isMLUsed() && !isMueLuUsed(),
    std::logic_error,
    "setCoordinates was called without setting an ML or MueLu parameter list.");

  if (isMLUsed()) {
    double *x, *y, *z;
    getCoordArrays(x, y, z);
    plist->set("x-coordinates", x);
    plist->set("y-coordinates", y);
    plist->set("z-coordinates", z);
    plist->set("PDE equations", numPDEs);
  } else {
    // Deep copy of the data.
    Teuchos::ArrayView<ST> xyzAV = Teuchos::arrayView(&xyz[0], xyz.size());
    Teuchos::RCP<Tpetra_MultiVector> xyzMV = Teuchos::rcp(
      new Tpetra_MultiVector(node_map, xyzAV, numNodes, numSpaceDim));
    plist->set("Coordinates", xyzMV);
    plist->set("number of equations", numPDEs);
  }  
}

void RigidBodyModes::
setCoordinatesAndNullspace(const Teuchos::RCP<const Tpetra_Map>& node_map,
                           const Teuchos::RCP<const Tpetra_Map>& soln_map)
{
  // numPDEs = # PDEs
  // numElasticityDim = # elasticity dofs
  // nullSpaceDim = dimension of elasticity nullspace
  // numScalar = # scalar dofs coupled to elasticity

  setCoordinates(node_map);

  if (numElasticityDim > 0 || setNonElastRBM == true ) {
    subtractCentroid(node_map, numSpaceDim, xyz);
    double *x, *y, *z;
    getCoordArrays(x, y, z);
    const LO numNodes = xyz.size() / numSpaceDim;
    if (setNonElastRBM == true) 
      Coord2RBM_nonElasticity(numNodes, x, y, z, nullSpaceDim, numScalar, nullSpaceDim, &rr[0]);
    else
      Coord2RBM(numNodes, x, y, z, numPDEs, numScalar, nullSpaceDim, &rr[0]);
    if (isMLUsed()) {
      plist->set("null space: type", "pre-computed");
      plist->set("null space: dimension", nullSpaceDim + numScalar);
      plist->set("null space: vectors", &rr[0]);
      plist->set("null space: add default vectors", false);     
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(
        soln_map.is_null(), std::logic_error,
        "numElasticityDim > 0 and isMueLuUsed(): soln_map must be provided.");
      Teuchos::ArrayView<ST> rrAV = Teuchos::arrayView(&rr[0], rr.size());
      Teuchos::RCP<Tpetra_MultiVector> Rbm = Teuchos::rcp(
        new Tpetra_MultiVector(soln_map, rrAV, soln_map->getNodeNumElements(),
                               nullSpaceDim + numScalar));
      plist->set("Nullspace", Rbm);
    }
  }
}
} // namespace Albany
