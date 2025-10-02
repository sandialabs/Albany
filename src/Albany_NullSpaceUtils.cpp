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
#include "Albany_Utils.hpp"

namespace Albany {

namespace {

// The null space is assumed to be formed by constants (translations) for each equations.
// In addition we can add rotation modes for physical vectors.
// We assume there is at most a vector of dimension vectorDim (either equal to 2 or 3),
// that is placed before the other PDEs
// The rotational modes are places after all the constant modes.
template <class Traits>
void ComputeNullSpace(
    Traits& nullSpace,
    const Teuchos::RCP<Thyra_MultiVector> &coordMV,
    int numPDEs, bool computeConstantModes,
    int physVectorDim, bool computeRotationModes)
{
  int numSpaceDim = coordMV->domain()->dim(); // Number of multivectors are the dimension of the problem
  auto data = getLocalData(coordMV.getConst());

  // At least component x should be there
  const int numNodes = data[0].size(); // local length of each vector

  Teuchos::ArrayRCP<const ST> x, y, z;
  x = data[0];
  if(numSpaceDim > 1) y = data[1];
  if(numSpaceDim > 2) z = data[2];

  nullSpace.zero();

  for (int node = 0 ; node < numNodes; node++) {

    int dof = node*numPDEs; //this is true only for interleaved order = 1

    int constModesOffset = computeConstantModes ? numPDEs : 0;

    //Constant (or translation) modes
    if(computeConstantModes) {
      for (int ii=0; ii<numPDEs; ii++)
        for (int jj=0; jj<numPDEs; jj++)
          nullSpace(dof, ii, jj) = (ii==jj) ? 1.0 : 0.0;
    }

    if(computeRotationModes) {
      if((physVectorDim >= 2) && (numSpaceDim > 1)) {
        /* xy rotation */
        nullSpace(dof, 0, constModesOffset) = -y[node];
        nullSpace(dof, 1, constModesOffset) = x[node];
      }

      if((physVectorDim == 3) && (numSpaceDim > 2)) {
        /* xz rotation */
        nullSpace(dof, 0, constModesOffset+1) = -z[node];
        nullSpace(dof, 2, constModesOffset+1) = x[node];

        /* yz rotation */
        nullSpace(dof, 1, constModesOffset+2) = -z[node];
        nullSpace(dof, 2, constModesOffset+2) = y[node];
      }
    }

  } /*for (node = 0 ; node < numNodes; node++)*/
} /*ComputeNullSpace*/

void subtractCentroid(const Teuchos::RCP<Thyra_MultiVector> &coordMV)
{
  auto spmd_vs = getSpmdVectorSpace(coordMV->range());
  const int numNodes = spmd_vs->localSubDim(); // local length of each vector
  const int ndim = coordMV->domain()->dim(); // Number of multivectors are the dimension of the problem

  auto data = getNonconstLocalData(coordMV);
  ST centroid[3]; // enough for up to 3d
  {
    ST sum[3];
    for (int i = 0; i < ndim; ++i) {
      Teuchos::ArrayRCP<const ST> x = data[i];
      sum[i] = 0;
      for (int j = 0; j < numNodes; ++j) sum[i] += x[j];
    }
    Teuchos::reduceAll(*createTeuchosCommFromThyraComm(spmd_vs->getComm()), Teuchos::REDUCE_SUM, ndim,
                                sum, centroid);
    const int numGlobalNodes = spmd_vs->dim(); // global length of each vector in the multivector
    for (int i = 0; i < ndim; ++i) centroid[i] /= numGlobalNodes;
  }

  for (int i = 0; i < ndim; ++i) {
    Teuchos::ArrayRCP<ST> x = data[i];
    for (Teuchos::ArrayRCP<ST>::size_type j = 0; j < numNodes; ++j)
      x[j] -= centroid[i];
  }
}

struct TpetraNullSpaceTraits {

  typedef Tpetra_MultiVector base_array_type;
  typedef Teuchos::RCP<base_array_type> array_type;
  array_type array;

  TpetraNullSpaceTraits(array_type array_) :
    array(array_) {}

  void zero(){
      array->putScalar(0.0);
  }

  double& operator()(const LO DOF, const int i, const int j){
     Teuchos::ArrayRCP<ST> rdata = array->getDataNonConst(j);
     return rdata[DOF + i];
  }
};

template <typename NullSpaceArray>
struct WriteNullSpace
{
  static void writeNullSpaceMethod(const NullSpaceArray& nullSpaceArray) {};
};

template<>
struct WriteNullSpace<Teuchos::RCP<const Tpetra_MultiVector>>
{
  static void writeNullSpaceMethod(const Teuchos::RCP<const Tpetra_MultiVector>& nullSpaceArray, const std::string name = "nullspace") {
    writeMatrixMarket(nullSpaceArray, name);
  };
};

} // namespace

// The base structure is empty. The derived one, stores an array,
// of the type specified by the Traits
// This struct allows us to hide tpetra info from the header file.
// When we decide what to use (Teko/MueLu/FROSch) we create a TraitsImpl
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
  typename Traits::array_type array;
};

RigidBodyModes::RigidBodyModes()
  : numPDEs(0),
    computeConstantModes(false),
    physVectorDim(0), computeRotationModes(false),
    nullSpaceDim(0),
    tekoUsed(false), mueLuUsed(false), froschUsed(false), setNonElastRBM(false),
    areProbParametersSet(false), arePiroParametersSet(false)
{}

void RigidBodyModes::
setPiroPL(const Teuchos::RCP<Teuchos::ParameterList>& piroParams)
{
  const Teuchos::RCP<Teuchos::ParameterList>
    stratList = Piro::extractStratimikosParams(piroParams);

  tekoUsed = mueLuUsed = froschUsed = false;
  if (Teuchos::nonnull(stratList) &&
      stratList->isParameter("Preconditioner Type")) {
    const std::string&
      ptype = stratList->get<std::string>("Preconditioner Type");
    if (ptype == "Teko") {
      plist = sublist(sublist(stratList, "Preconditioner Types", true), ptype, true);
      tekoUsed = true;
    }
    else if (ptype == "MueLu") {
      plist = sublist(sublist(stratList, "Preconditioner Types", true), ptype, true);
      mueLuUsed = true;
    }
    else if (ptype == "FROSch") {
      plist = sublist(sublist(stratList, "Preconditioner Types", true), ptype, true);
      froschUsed = true;
    }
  }

  if (tekoUsed) {
    // Two nullspace vectors. One for UV, one for W and T
    nullSpaceTraits.push_back(Teuchos::rcp(new TraitsImpl<TpetraNullSpaceTraits>()));
    nullSpaceTraits.push_back(Teuchos::rcp(new TraitsImpl<TpetraNullSpaceTraits>()));
  }
  else {
    nullSpaceTraits.push_back(Teuchos::rcp(new TraitsImpl<TpetraNullSpaceTraits>()));
  }

  arePiroParametersSet = true;
}

void RigidBodyModes::
updatePL(const Teuchos::RCP<Teuchos::ParameterList>& precParams)
{
  plist = precParams;
}

void RigidBodyModes::setParameters(
  const int numPDEs_, const bool computeConstantModes_ ,
  const int physVectorDim_, const bool computeRotationModes_)
{
  numPDEs = numPDEs_;
  computeConstantModes = computeConstantModes_;
  physVectorDim = physVectorDim_;
  computeRotationModes = computeRotationModes_;

  //the number of constant modes equals numPDEs
  nullSpaceDim = computeConstantModes_ ? numPDEs_ : 0;

  //for 2d vector we add x-y rotations,
  //for 3d vector we add x-y, x-z, y-z rotations
  if(computeRotationModes_)
    nullSpaceDim += (physVectorDim_==2) + 3*(physVectorDim_==3);

  areProbParametersSet = true;
}

void RigidBodyModes::
setCoordinates(const Teuchos::RCP<Thyra_MultiVector>& coordMV_)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      !areProbParametersSet,
      std::logic_error,
      "RigidBodyModes::setCoordinates was called before calling RigidBodyModes::setParameters\n" <<
      "RigidBodyModes::setParameters should be called by the problem constructor.");

  TEUCHOS_TEST_FOR_EXCEPTION(
      !arePiroParametersSet,
      std::logic_error,
      "RigidBodyModes::setCoordinates was called before calling RigidBodyModes::setPiroPL.");

  TEUCHOS_TEST_FOR_EXCEPTION(
    !tekoUsed && !mueLuUsed && !froschUsed,
    std::logic_error,
    "setCoordinates was called without setting a Teko, MueLu or FROSch parameter list.");
  
  coordMV = coordMV_;

  if (tekoUsed) {  // Teko here
    auto t_coordMV = getTpetraMultiVector(coordMV);
    auto invLibList = sublist(plist, "Inverse Factory Library", true);

    auto mueluUVList = sublist(invLibList, "myMueLuUV", true);
    mueluUVList->set("Coordinates", t_coordMV);

    auto mueluTList = sublist(invLibList, "myMueLuT", true);
    mueluTList->set("Coordinates", t_coordMV);

    if (invLibList->isSublist("myMueLuW")) {
      auto mueluWList = sublist(invLibList, "myMueLuW", true);
      mueluWList->set("Coordinates", t_coordMV);
    }
  } else if (mueLuUsed) {  // MueLu here
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
setCoordinatesAndComputeNullspace(const Teuchos::RCP<Thyra_MultiVector>& coordMV_in,
                           const Teuchos::RCP<const Thyra_VectorSpace>& soln_vs,
                           const Teuchos::RCP<const Thyra_VectorSpace>& soln_overlap_vs)
{
  setCoordinates(coordMV_in);

  if (nullSpaceDim > 0) {

    subtractCentroid(coordMV);

    {  // Teko, MueLu and FROSch
      TEUCHOS_TEST_FOR_EXCEPTION(soln_vs.is_null(), std::logic_error,
          "nullSpaceDim > 0 and (tekoUsed or mueLuUsed or froschUsed): solution vector space must be provided.");

      using NullSpaceTraits = TpetraNullSpaceTraits;
      if (tekoUsed) {
        auto invLibList = sublist(plist, "Inverse Factory Library", true);

        auto& tpetraTraitsArrayUV =
            Teuchos::rcp_dynamic_cast<TraitsImpl<NullSpaceTraits>>(nullSpaceTraits[0])->array;
        tpetraTraitsArrayUV =
            Teuchos::rcp(new NullSpaceTraits::base_array_type(getTpetraMap(soln_vs), nullSpaceDim-2, false));
        NullSpaceTraits nullSpaceUV(tpetraTraitsArrayUV);
        ComputeNullSpace(nullSpaceUV, coordMV, 2, computeConstantModes, physVectorDim, computeRotationModes);
        auto mueluUVList = sublist(invLibList, "myMueLuUV", true);
        mueluUVList->set("Nullspace", tpetraTraitsArrayUV);

        auto& tpetraTraitsArrayT =
            Teuchos::rcp_dynamic_cast<TraitsImpl<NullSpaceTraits>>(nullSpaceTraits[1])->array;
        tpetraTraitsArrayT =
            Teuchos::rcp(new NullSpaceTraits::base_array_type(getTpetraMap(soln_vs), 1, false));
        NullSpaceTraits nullSpaceT(tpetraTraitsArrayT);
        ComputeNullSpace(nullSpaceT, coordMV, 1, computeConstantModes, 1, false);
        auto mueluTList = sublist(invLibList, "myMueLuT", true);
        mueluTList->set("Nullspace", tpetraTraitsArrayT);

        if (invLibList->isSublist("myMueLuW")) {
          auto mueluWList = sublist(invLibList, "myMueLuW", true);
          mueluWList->set("Nullspace", tpetraTraitsArrayT);
        }
      }
      else {
        auto& tpetraTraitsArray =
            Teuchos::rcp_dynamic_cast<TraitsImpl<NullSpaceTraits>>(nullSpaceTraits[0])->array;
        tpetraTraitsArray =
            Teuchos::rcp(new NullSpaceTraits::base_array_type(getTpetraMap(soln_vs), nullSpaceDim, false));
        NullSpaceTraits nullSpace(tpetraTraitsArray);
        ComputeNullSpace(nullSpace, coordMV, numPDEs, computeConstantModes, physVectorDim, computeRotationModes);
        if (mueLuUsed) {
          plist->set("Nullspace", tpetraTraitsArray);
          // WriteNullSpace<Teuchos::RCP<const NullSpaceTraits::base_array_type>>::writeNullSpaceMethod(tpetraTraitsArray);
        } else { // This means that FROSch is used
          plist->set("Null Space", tpetraTraitsArray);
        }
      }
    }
  }
  if(froschUsed) {
    TEUCHOS_TEST_FOR_EXCEPTION(
      soln_overlap_vs.is_null(), std::logic_error,
      "froschUsed: soln_overlap_map must be provided.");
    plist->set("Repeated Map",getTpetraMap(soln_overlap_vs));
  }
}

} // namespace Albany
