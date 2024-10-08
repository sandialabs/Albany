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
  ebName         = "";
}

MeshSpecsStruct::
MeshSpecsStruct(const MeshType           type,
                const CellTopologyData&  ctd_,
                int                      numDim_,
                std::vector<std::string> nsNames_,
                std::vector<std::string> ssNames_,
                int                      worksetSize_,
                const std::string        ebName_,
                std::map<std::string, int> ebNameToIndex_)
 : mesh_type(type)
 , ctd(ctd_)
 , numDim(numDim_)
 , nsNames(nsNames_)
 , ssNames(ssNames_)
 , worksetSize(worksetSize_)
 , ebName(ebName_)
 , ebNameToIndex(ebNameToIndex_)
{
  // Nothing to do here
}

} // namespace Albany
