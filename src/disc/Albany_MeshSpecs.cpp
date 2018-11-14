#include "Albany_MeshSpecs.hpp"

#include "Teuchos_TestForException.hpp"
#include "Shards_CellTopologyTraits.hpp"
#include "Shards_BasicTopologies.hpp"

namespace Albany {

MeshSpecsStruct::MeshSpecsStruct()
{
  ctd.name       = "NULL";
  numDim         = -1;
  worksetSize    = -1;
  cubatureDegree = -1;
  ebName         = "";
}

MeshSpecsStruct::MeshSpecsStruct(
    const CellTopologyData&  ctd_,
    int                      numDim_,
    int                      cubatureDegree_,
    std::vector<std::string> nsNames_,
    std::vector<std::string> ssNames_,
    int                      worksetSize_,
    const std::string        ebName_,
    std::map<std::string, int> ebNameToIndex_,
    bool                       interleavedOrdering_,
    const bool                 sepEvalsByEB_,
    const Intrepid2::EPolyType cubatureRule_)
    : ctd(ctd_),
      numDim(numDim_),
      cubatureDegree(cubatureDegree_),
      nsNames(nsNames_),
      ssNames(ssNames_),
      worksetSize(worksetSize_),
      ebName(ebName_),
      ebNameToIndex(ebNameToIndex_),
      interleavedOrdering(interleavedOrdering_),
      sepEvalsByEB(sepEvalsByEB_),
      cubatureRule(cubatureRule_)
{
  TEUCHOS_TEST_FOR_EXCEPTION (cubatureDegree<0, Teuchos::Exceptions::InvalidArgument,
                              "Error! Invalid cubature degree on element block '" << ebName << "'.\n");
}

} // namespace Albany
