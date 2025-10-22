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

int ComputeNullSpaceDim(
  const int numPDEs_, const bool computeConstantModes_,
  const int physVectorDim_, const bool computeRotationModes_)
{
  //the number of constant modes equals numPDEs
  int nullSpaceDim_ = computeConstantModes_ ? numPDEs_ : 0;

  //for 2d vector we add x-y rotations,
  //for 3d vector we add x-y, x-z, y-z rotations
  if(computeRotationModes_)
    nullSpaceDim_ += (physVectorDim_==2) + 3*(physVectorDim_==3);
  return nullSpaceDim_;
}

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
    // Read block decomposition
    std::stringstream ss;
    ss << plist->get<std::string>("Strided Blocking");
    while (not ss.eof()) {
      int num = 0;
      ss >> num;
      TEUCHOS_ASSERT(num > 0);
      tekoBlockDecomp.push_back(num);
    }
    const int numBlocks = tekoBlockDecomp.size();

    // Read block preconditioner parameter lists
    const auto& invType = plist->get<std::string>("Inverse Type");
    auto invLibList = sublist(plist, "Inverse Factory Library", true);
    auto invList = sublist(invLibList, invType, true);
    for (int blk = 0; blk < numBlocks; ++blk) {
      auto blkInvTypeKey = std::string("Inverse Type ") + std::to_string(blk+1);
      const auto& blkInvType = invList->get<std::string>(blkInvTypeKey);
      tekoBlockPlists.push_back(sublist(invLibList, blkInvType, true));
    }

    // Create nullspace vectors
    for (int blk = 0; blk < numBlocks; ++blk) {
      nullSpaceTraits.push_back(Teuchos::rcp(new TraitsImpl<TpetraNullSpaceTraits>()));
    }
  }
  else {
    nullSpaceTraits.push_back(Teuchos::rcp(new TraitsImpl<TpetraNullSpaceTraits>()));
  }

  arePiroParametersSet = true;
}

void RigidBodyModes::setParameters(
  const int numPDEs_, const bool computeConstantModes_ ,
  const int physVectorDim_, const bool computeRotationModes_)
{
  numPDEs = numPDEs_;
  computeConstantModes = computeConstantModes_;
  physVectorDim = physVectorDim_;
  computeRotationModes = computeRotationModes_;
  nullSpaceDim = ComputeNullSpaceDim(numPDEs_, computeConstantModes_, physVectorDim_, computeRotationModes_);

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
    // Add coords if muelu/frosch block
    auto t_coordMV = getTpetraMultiVector(coordMV);
    const int numBlocks = tekoBlockDecomp.size();
    for (int blk = 0; blk < numBlocks; ++blk) {
      auto blkPlist = tekoBlockPlists[blk];
      const auto& blkPrecType = blkPlist->get<std::string>("Type");
      if (blkPrecType == "MueLu") {
        blkPlist->set("Coordinates", t_coordMV);
      }
      else if (blkPrecType == "FROSch") {
        blkPlist->set("Coordinates List", t_coordMV);
      }
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
  TEUCHOS_TEST_FOR_EXCEPTION(
      soln_vs.is_null(), std::logic_error,
      "solution vector space must be provided.");
  TEUCHOS_TEST_FOR_EXCEPTION(
      soln_overlap_vs.is_null(), std::logic_error,
      "soln_overlap_map must be provided.");

  setCoordinates(coordMV_in);

  // Compute/set nullspace
  if (nullSpaceDim > 0) {
    subtractCentroid(coordMV);
    if (tekoUsed) {
      const int numBlocks = tekoBlockDecomp.size();
      const int numEqns = numPDEs;
      const auto numDofs = Albany::getLocalSubdim(soln_vs);
      TEUCHOS_ASSERT(numDofs % numEqns == 0);
      const LO numDofsPerEqn = numDofs / numEqns;
      for (int blk = 0; blk < numBlocks; ++blk) {
        // Add nullspace if muelu/frosch block
        auto blkPlist = tekoBlockPlists[blk];
        const auto& blkPrecType = blkPlist->get<std::string>("Type");
        if (blkPrecType != "MueLu" and blkPrecType != "FROSch")
          continue;

        // Create tpetra map
        const int blkNumEqns = tekoBlockDecomp[blk];
        const int blkNumGIDs = blkNumEqns * numDofsPerEqn;
        auto tpetraMap = Teuchos::rcp(new Tpetra_Map(Teuchos::OrdinalTraits<GO>::invalid(), blkNumGIDs, 0, Albany::getComm(soln_vs)));

        // Allocate/compute nullspace
        const int blkPhysVectorDim = blkNumEqns < physVectorDim ? blkNumEqns : physVectorDim;
        const int blkNullSpaceDim = ComputeNullSpaceDim(blkNumEqns, computeConstantModes, blkPhysVectorDim, computeRotationModes);
        auto& tpetraTraitsArray =
            Teuchos::rcp_dynamic_cast<TraitsImpl<TpetraNullSpaceTraits>>(nullSpaceTraits[blk])->array;
        tpetraTraitsArray =
            Teuchos::rcp(new TpetraNullSpaceTraits::base_array_type(tpetraMap, blkNullSpaceDim, false));
        TpetraNullSpaceTraits nullSpace(tpetraTraitsArray);
        ComputeNullSpace(nullSpace, coordMV, blkNumEqns, computeConstantModes, blkPhysVectorDim, computeRotationModes);
        if (blkPrecType == "MueLu") {
          blkPlist->set("Nullspace", tpetraTraitsArray);
        }
        else if (blkPrecType == "FROSch") {
          blkPlist->set("Null Space", tpetraTraitsArray);
        }
      }
    }
    else {
      auto& tpetraTraitsArray =
          Teuchos::rcp_dynamic_cast<TraitsImpl<TpetraNullSpaceTraits>>(nullSpaceTraits[0])->array;
      tpetraTraitsArray =
          Teuchos::rcp(new TpetraNullSpaceTraits::base_array_type(getTpetraMap(soln_vs), nullSpaceDim, false));
      TpetraNullSpaceTraits nullSpace(tpetraTraitsArray);
      ComputeNullSpace(nullSpace, coordMV, numPDEs, computeConstantModes, physVectorDim, computeRotationModes);
      if (mueLuUsed) {
        plist->set("Nullspace", tpetraTraitsArray);
      }
      else if (froschUsed) {
        plist->set("Null Space", tpetraTraitsArray);
      }
    }
  }

  // Add repeated map
  // TODO: Teko - Passing a "Repeated Map" of a block throws a segfault. This doesn't seem to be needed though. Consider removing all together.
  if (froschUsed) {
    plist->set("Repeated Map",getTpetraMap(soln_overlap_vs));
  }
}

} // namespace Albany
