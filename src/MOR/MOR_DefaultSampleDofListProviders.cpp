//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_DefaultSampleDofListProviders.hpp"

#include "MOR_EpetraUtils.hpp"

#include "Epetra_BlockMap.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

namespace MOR {

AllSampleDofListProvider::AllSampleDofListProvider(const Teuchos::RCP<const Epetra_BlockMap> &map) :
  map_(map)
{
  // Nothing to do
}

Teuchos::Array<int>
AllSampleDofListProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &/*params*/)
{
  Teuchos::Array<int> result;
  const int entryCount = map_->NumMyElements();
  result.reserve(entryCount);
  for (int i = 0; i < entryCount; ++i) {
    result.push_back(i);
  }
  return result;
}

namespace { // anonymous

inline
Teuchos::Array<int>
getDofListFromParam(const Teuchos::ParameterList &params)
{
  return Teuchos::getParameter<Teuchos::Array<int> >(params, "Sample Dof List");
}

} // end anonymous namespace

InlineSampleDofListProvider::InlineSampleDofListProvider(const Teuchos::RCP<const Epetra_BlockMap> &map) :
  map_(map)
{
  // Nothing to do
}

Teuchos::Array<int>
InlineSampleDofListProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  return getMyLIDs(*map_, getDofListFromParam(*params));
}

XMLFileSampleDofListProvider::XMLFileSampleDofListProvider(const Teuchos::RCP<const Epetra_BlockMap> &map) :
  map_(map)
{
  // Nothing to do
}

Teuchos::Array<int>
XMLFileSampleDofListProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string path = Teuchos::getParameter<std::string>(*params, "Sample Dof Input File Name");
  const Teuchos::RCP<const Teuchos::ParameterList> sampleDofsParams = Teuchos::getParametersFromXmlFile(path);
  return getMyLIDs(*map_, getDofListFromParam(*sampleDofsParams));
}

Teuchos::RCP<SampleDofListFactory> defaultSampleDofListFactoryNew(const Teuchos::RCP<const Epetra_BlockMap> &map)
{
  const Teuchos::RCP<SampleDofListFactory> result(new SampleDofListFactory);

  result->extend("All", Teuchos::rcp(new AllSampleDofListProvider(map)));
  result->extend("Inline", Teuchos::rcp(new InlineSampleDofListProvider(map)));
  result->extend("File", Teuchos::rcp(new XMLFileSampleDofListProvider(map)));

  return result;
}

} // namespace MOR
