//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Piro_StratimikosUtils.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "Albany_NullSpaceUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"

namespace Albany {

namespace {

// Copied from Trilinos/packages/ml/src/Utils/ml_rbm.c.
template <class Traits>
void Coord2RBM(
  const Teuchos::RCP<Thyra_MultiVector> &coordMV,
  const int Ndof, const int NscalarDof, const int NSdim,
  typename Traits::array_type &rbm)
{
  int ii, jj;
  int dof;

  const int numSpaceDim = coordMV->domain()->dim(); // Number of multivectors are the dimension of the problem

  auto data = getLocalData(coordMV.getConst());
  const int numNodes = data[0].size(); // length of each vector in the multivector

  const int vec_leng = numNodes*Ndof;
  Traits traits_class(Ndof, NscalarDof, NSdim, vec_leng, rbm);

  Teuchos::ArrayRCP<const ST> x = data[0];
  Teuchos::ArrayRCP<const ST> y, z;
  if(numSpaceDim > 1) {
      y = data[1];
  }
  if(numSpaceDim > 2) {
      z = data[2];
  }

  traits_class.zero();

  for (int node = 0 ; node < numNodes; node++) {
    dof = node*Ndof;
    switch( Ndof - NscalarDof ) {
    case 6:
      for (ii=3;ii<6+NscalarDof;ii++) { /* lower half = [ 0 I ] */
        for (jj=0;jj<6+NscalarDof;jj++) {
          traits_class.ArrObj(dof, ii, jj) = (ii==jj) ? 1.0 : 0.0;
        }
      }
      /* There is no break here and that is on purpose */
    case 3:
      for (ii=0;ii<3+NscalarDof;ii++) { /* upper left = [ I ] */
        for (jj=0;jj<3+NscalarDof;jj++) {
          traits_class.ArrObj(dof, ii, jj) = (ii==jj) ? 1.0 : 0.0;
        }
      }
      for (ii=0;ii<3;ii++) { /* upper right = [ Q ] */
        for (jj=3+NscalarDof;jj<6+NscalarDof;jj++) {
          // std::cout <<"jj " << jj << " " << ii + jj << std::endl;
          if(ii == jj-3-NscalarDof) traits_class.ArrObj(dof, ii, jj) = 0.0;
          else {
            if (ii+jj == 4+NscalarDof) traits_class.ArrObj(dof, ii, jj) = z[node];
            else if ( ii+jj == 5+NscalarDof ) traits_class.ArrObj(dof, ii, jj) = y[node];
            else if ( ii+jj == 6+NscalarDof ) traits_class.ArrObj(dof, ii, jj) = x[node];
            else traits_class.ArrObj(dof, ii, jj) = 0.0;
          }
        }
      }
      ii = 0; jj = 5+NscalarDof; traits_class.ArrObj(dof, ii, jj) *= -1.0;
      ii = 1; jj = 3+NscalarDof; traits_class.ArrObj(dof, ii, jj) *= -1.0;
      ii = 2; jj = 4+NscalarDof; traits_class.ArrObj(dof, ii, jj) *= -1.0;
      break;

    case 2:
      for (ii=0;ii<2+NscalarDof;ii++) { /* upper left = [ I ] */
        for (jj=0;jj<2+NscalarDof;jj++) {
          traits_class.ArrObj(dof, ii, jj) = (ii==jj) ? 1.0 : 0.0;
        }
      }
      for (ii=0;ii<2+NscalarDof;ii++) { /* upper right = [ Q ] */
        for (jj=2+NscalarDof;jj<3+NscalarDof;jj++) {
          if (ii == 0) traits_class.ArrObj(dof, ii, jj) = -y[node];
          else {
            if (ii == 1) { traits_class.ArrObj(dof, ii, jj) =  x[node];}
            else traits_class.ArrObj(dof, ii, jj) = 0.0;
          }
        }
      }
      break;

    case 1:
      for (ii = 0; ii<1+NscalarDof; ii++) {
        for (jj=0; jj<1+NscalarDof; jj++) {
          traits_class.ArrObj(dof, ii, jj) = (ii == jj) ? 1.0 : 0.0;
        }
      }
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Coord2RBM: Ndof = " << Ndof << " not implemented\n");
    } /*switch*/

  } /*for (node = 0 ; node < numNodes; node++)*/

  return;
} /*Coord2RBM*/

//IKT, 6/28/15: the following set RBMs for non-elasticity problems.
template <class Traits>
void Coord2RBM_nonElasticity(
  const Teuchos::RCP<Thyra_MultiVector> &coordMV,
  const int Ndof, const int NscalarDof, const int NSdim,
  typename Traits::array_type &rbm)
{
  //std::cout << "setting RBMs in Coord2RBM_nonElasticity!" << std::endl;
  int ii, jj;
  int dof;

  int numSpaceDim = coordMV->domain()->dim(); // Number of multivectors are the dimension of the problem
  auto data = getLocalData(coordMV.getConst());

  // At least component x should be there
  const int numNodes = data[0].size(); // length of each vector in the multivector

  const int vec_leng = numNodes*Ndof;
  Traits traits_class(Ndof, NscalarDof, NSdim, vec_leng, rbm);

  Teuchos::ArrayRCP<const ST> x = data[0];
  Teuchos::ArrayRCP<const ST> y, z;
  if(numSpaceDim > 1) {
      y = data[1];
  }
  if(numSpaceDim > 2) {
      z = data[2];
  }

  traits_class.zero();

  //std::cout << "...Ndof: " << Ndof << std::endl;
  //std::cout << "...case: " << NSdim - NscalarDof << std::endl;
  for (int node = 0 ; node < numNodes; node++) {

    dof = node*Ndof;

    switch( NSdim - NscalarDof ) {
    case 3:
      for (ii=0;ii<2;ii++) { /* upper right = [ Q ] -- xy rotation only */
        jj = 2+NscalarDof;
        // std::cout <<"jj " << jj << " " << ii + jj << std::endl;
        if (ii == 0)
          traits_class.ArrObj(dof, ii, jj) = y[node];
        else if (ii == 1)
          traits_class.ArrObj(dof, ii, jj) = x[node];
      }
      ii = 0; jj = 2+NscalarDof;
      traits_class.ArrObj(dof, ii, jj) *= -1.0;
      /* There is no break here and that is on purpose */
    case 2:
      for (ii=0;ii<2+NscalarDof;ii++) { /* upper left = [ I ] */
        for (jj=0;jj<2+NscalarDof;jj++) {
          traits_class.ArrObj(dof, ii, jj) = (ii==jj) ? 1.0 : 0.0;
        }
      }
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "Coord2RBM_nonElasticity: Ndof = " << Ndof << " not implemented\n");
    } /*switch*/

  } /*for (node = 0 ; node < numNodes; node++)*/

} /*Coord2RBM_nonElasticity*/

void subtractCentroid(const Teuchos::RCP<Thyra_MultiVector> &coordMV)
{
  auto spmd_vs = getSpmdVectorSpace(coordMV->range());
  const int nnodes = spmd_vs->localSubDim(); // local length of each vector
  const int ndim = coordMV->domain()->dim(); // Number of multivectors are the dimension of the problem

  auto data = getNonconstLocalData(coordMV);
  ST centroid[3]; // enough for up to 3d
  {
    ST sum[3];
    for (int i = 0; i < ndim; ++i) {
      Teuchos::ArrayRCP<const ST> x = data[i];
      sum[i] = 0;
      for (int j = 0; j < nnodes; ++j) sum[i] += x[j];
    }
    Teuchos::reduceAll(*createTeuchosCommFromThyraComm(spmd_vs->getComm()), Teuchos::REDUCE_SUM, ndim,
                                sum, centroid);
    const int numNodes = spmd_vs->localSubDim(); // length of each vector in the multivector
    for (int i = 0; i < ndim; ++i) centroid[i] /= numNodes;
  }

  for (int i = 0; i < ndim; ++i) {
    Teuchos::ArrayRCP<ST> x = data[i];
    for (Teuchos::ArrayRCP<ST>::size_type j = 0; j < nnodes; ++j)
      x[j] -= centroid[i];
  }
}

struct Tpetra_NullSpace_Traits {

  typedef Tpetra_MultiVector base_array_type;
  typedef Teuchos::RCP<base_array_type> array_type;
  const int Ndof;
  const int NscalarDof;
  const int NSdim;
  const LO vec_leng;
  array_type Array;

  Tpetra_NullSpace_Traits(const int ndof, const int nscalardof, const int nsdim,
     const LO veclen, array_type &array)
   : Ndof(ndof), NscalarDof(nscalardof), NSdim(nsdim), vec_leng(veclen), Array(array) {}

  void zero(){
      Array->putScalar(0.0);
  }

  double &ArrObj(const LO DOF, const int i, const int j){
     Teuchos::ArrayRCP<ST> rdata = Array->getDataNonConst(j);
     return rdata[DOF + i];
  }

};

struct Epetra_NullSpace_Traits {

  typedef std::vector<ST> array_type;
  const int Ndof;
  const int NscalarDof;
  const int NSdim;
  const array_type::size_type vec_leng;
  array_type& Array;

  Epetra_NullSpace_Traits(const int ndof, const int nscalardof, const int nsdim, const array_type::size_type veclen,
      array_type &array)
   : Ndof(ndof), NscalarDof(nscalardof), NSdim(nsdim), vec_leng(veclen), Array(array) {}

  void zero(){
    for (array_type::size_type i = 0; i < vec_leng*(NSdim + NscalarDof); i++)
       Array[i] = 0.0;
  }

  double &ArrObj(const array_type::size_type DOF, const int i, const int j){
     return Array[DOF + i + j * vec_leng];
  }

};

} // namespace

// The base structure is empty. The derived one, stores an array,
// of the type specified by the Traits
// This struct allows us to hide tpetra/epetra info from the header file.
// When we decide what to use (ML or MueLu) we create a TraitsImpl
// templated on the 'actual' traits, and we grab the array.
// This way the null space stores a persistent array (being stored inside
// the class makes sure the array does not disappear like it would
// if it were a temporary), and yet the class need not to know what
// kind of array it is.
struct TraitsImplBase {
  virtual ~TraitsImplBase () = default;
};

template<typename Traits>
struct TraitsImpl : public TraitsImplBase {
  typename Traits::array_type arr;
};


RigidBodyModes::RigidBodyModes(int numPDEs_)
  : numPDEs(numPDEs_), numElasticityDim(0), numScalar(0), nullSpaceDim(0),
    mlUsed(false), mueLuUsed(false), froschUsed(false), setNonElastRBM(false)
{}

void RigidBodyModes::
setPiroPL(const Teuchos::RCP<Teuchos::ParameterList>& piroParams)
{
  const Teuchos::RCP<Teuchos::ParameterList>
    stratList = Piro::extractStratimikosParams(piroParams);

  mlUsed = mueLuUsed = froschUsed = false;
  if (Teuchos::nonnull(stratList) &&
      stratList->isParameter("Preconditioner Type")) {
    const std::string&
      ptype = stratList->get<std::string>("Preconditioner Type");
    if (ptype == "ML") {
      plist = sublist(sublist(sublist(stratList, "Preconditioner Types"),
                              ptype), "ML Settings");
      mlUsed = true;
    }
    else if (ptype == "MueLu") {
      plist = sublist(sublist(stratList, "Preconditioner Types"), ptype);
      mueLuUsed = true;
    }
    else if (ptype == "FROSch") {
      plist = sublist(sublist(stratList, "Preconditioner Types"), ptype);
      froschUsed = true;
    }
  } 

  if (mlUsed) {
    traits = Teuchos::rcp( new TraitsImpl<Epetra_NullSpace_Traits>());
  } else {
    traits = Teuchos::rcp( new TraitsImpl<Tpetra_NullSpace_Traits>());
  }
}

void RigidBodyModes::
updatePL(const Teuchos::RCP<Teuchos::ParameterList>& precParams)
{
  plist = precParams;
}

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
setCoordinates(const Teuchos::RCP<Thyra_MultiVector>& coordMV_)
{
  coordMV = coordMV_;

  TEUCHOS_TEST_FOR_EXCEPTION(
    !isMLUsed() && !isMueLuUsed() && !isFROSchUsed(),
    std::logic_error,
    "setCoordinates was called without setting an ML, MueLu or FROSch parameter list.");

  int numSpaceDim = coordMV->domain()->dim(); // Number of multivectors are the dimension of the problem

  if (isMLUsed()) { // ML here

    //MP: Even when a processor has no nodes, ML requires a nonnull pointer of coordinates.
    double emptyCoords[3] ={0.0, 0.0, 0.0};

    double* x = getNonconstLocalData(coordMV->col(0)).getRawPtr();

    if (x==nullptr) {
      x = &emptyCoords[0];
    }
    plist->set<double*>("x-coordinates", x);
    if(numSpaceDim > 1){
      double* y = getNonconstLocalData(coordMV->col(1)).getRawPtr();
      if (y==nullptr) {
        y = &emptyCoords[1];
      }
      plist->set<double*>("y-coordinates", y);
    } else {
      plist->set<double*>("y-coordinates", nullptr);
    }
    if(numSpaceDim > 2){
      double* z = getNonconstLocalData(coordMV->col(2)).getRawPtr();
      if (z==nullptr) {
        z = &emptyCoords[2];
      }
      plist->set<double*>("z-coordinates", z);
    } else {
      plist->set<double*>("z-coordinates", nullptr);
    }

    plist->set("PDE equations", numPDEs);

  } else if (isMueLuUsed()) {  // MueLu here
    // It apperas MueLu only accepts Tpetra. Get the Tpetra MV then.
    auto t_coordMV = getTpetraMultiVector(coordMV);
    if (plist->isSublist("Factories") == true) {
      // use verbose input deck
      Teuchos::ParameterList& matrixList = plist->sublist("Matrix");
      matrixList.set("PDE equations", numPDEs);
      plist->set("Coordinates", t_coordMV);
    } else {
      // use simplified input deck
      plist->set("Coordinates", t_coordMV);
      plist->set("number of equations", numPDEs);
    }
  } else { // FROSch here
    auto t_coordMV = getTpetraMultiVector(coordMV);
    plist->set("Coordinates List",t_coordMV);
  }
}

void RigidBodyModes::
setCoordinatesAndNullspace(const Teuchos::RCP<Thyra_MultiVector>& coordMV_in,
                           const Teuchos::RCP<const Thyra_VectorSpace>& soln_vs,
                           const Teuchos::RCP<const Thyra_VectorSpace>& soln_overlap_vs)
{
  setCoordinates(coordMV_in);

  // numPDEs = # PDEs
  // numElasticityDim = # elasticity dofs
  // nullSpaceDim = dimension of elasticity nullspace
  // numScalar = # scalar dofs coupled to elasticity

  int numSpaceDim = coordMV->domain()->dim(); // Number of multivectors are the dimension of the problem
  const int numNodes = getSpmdVectorSpace(coordMV->range())->localSubDim();

  if (numElasticityDim > 0 || setNonElastRBM == true ) {

    if (isMLUsed()) {

      using Traits = Epetra_NullSpace_Traits;
      auto e_traits = Teuchos::rcp_dynamic_cast<TraitsImpl<Traits>>(traits);
      auto& err = e_traits->arr;

      if(nullSpaceDim > 0) {
        if (setNonElastRBM == true) {
          err.resize((nullSpaceDim + numScalar) * numSpaceDim * numNodes);
        }
        else {
          err.resize((nullSpaceDim + numScalar) * numPDEs * numNodes);
        }
      }

      subtractCentroid(coordMV);

      if (setNonElastRBM == true)
        Coord2RBM_nonElasticity<Epetra_NullSpace_Traits>(coordMV, numPDEs, numScalar, nullSpaceDim, err);
      else
        Coord2RBM<Epetra_NullSpace_Traits>(coordMV, numPDEs, numScalar, nullSpaceDim, err);

      plist->set("null space: type", "pre-computed");
      plist->set("null space: dimension", nullSpaceDim + numScalar);
      plist->set("null space: vectors", &err[0]);
      plist->set("null space: add default vectors", false);

    } else {  // MueLu and FROSch
      using Traits = Tpetra_NullSpace_Traits;
      auto t_traits = Teuchos::rcp_dynamic_cast<TraitsImpl<Traits>>(traits);
      auto& trr = t_traits->arr;
      trr = Teuchos::rcp(new Tpetra_NullSpace_Traits::base_array_type(getTpetraMap(soln_vs),
                               nullSpaceDim + numScalar, false));

      subtractCentroid(coordMV);

      if (setNonElastRBM == true)
        Coord2RBM_nonElasticity<Tpetra_NullSpace_Traits>(coordMV, numPDEs, numScalar, nullSpaceDim, trr);
      else
        Coord2RBM<Tpetra_NullSpace_Traits>(coordMV, numPDEs, numScalar, nullSpaceDim, trr);

      TEUCHOS_TEST_FOR_EXCEPTION(
        soln_vs.is_null(), std::logic_error,
        "numElasticityDim > 0 and (isMueLuUsed() or isFROSchUsed()): soln_map must be provided.");
        if (isMueLuUsed()) {
          plist->set("Nullspace", trr);
        } else { // This means that FROSch is used
          plist->set("Null Space",trr);
        }
    }
  }
  if(isFROSchUsed()) {
    TEUCHOS_TEST_FOR_EXCEPTION(
      soln_overlap_vs.is_null(), std::logic_error,
      "isFROSchUsed(): soln_overlap_map must be provided.");
    plist->set("Repeated Map",getTpetraMap(soln_overlap_vs));
  }
}

} // namespace Albany
