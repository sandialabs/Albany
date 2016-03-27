//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DiscretizationDofListProvider.hpp"

#include "Teuchos_Array.hpp"

#include "Teuchos_TestForException.hpp"

#include <vector>
#include <string>

namespace Albany {

DiscretizationSampleDofListProvider::DiscretizationSampleDofListProvider(
    const Teuchos::RCP<const AbstractDiscretization> &disc) :
  disc_(disc)
{
  // Nothing to do
}

Teuchos::Array<int>
DiscretizationSampleDofListProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &/*params*/)
{
  Teuchos::Array<int> result;
  {
    const NodeSetList &allNodeSets = disc_->getNodeSets();

    const std::string nodeSetName = "sample_nodes";
    const NodeSetList::const_iterator it = allNodeSets.find(nodeSetName);
    TEUCHOS_TEST_FOR_EXCEPTION(
        it == allNodeSets.end(),
        std::out_of_range,
        "Nodeset " << nodeSetName << " does not exist");

    typedef NodeSetList::mapped_type NodeSetEntryList;
    const NodeSetEntryList &sampleNodeEntries = it->second;
    for (NodeSetEntryList::const_iterator jt = sampleNodeEntries.begin(); jt != sampleNodeEntries.end(); ++jt) {
      typedef NodeSetEntryList::value_type NodeEntryList;
      const NodeEntryList &sampleEntries = *jt;
      for (NodeEntryList::const_iterator kt = sampleEntries.begin(); kt != sampleEntries.end(); ++kt) {
        result.push_back(*kt);
      }
    }
  }
  return result;
}

} // end namespace Albany
