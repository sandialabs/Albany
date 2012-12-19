//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_DefaultSampleDofListProviders.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

namespace Albany {

AllSampleDofListProvider::AllSampleDofListProvider(int dofCount) :
  dofCount_(dofCount)
{
  // Nothing to do
}

Teuchos::Array<int>
AllSampleDofListProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &/*params*/)
{
  Teuchos::Array<int> result;
  result.reserve(dofCount_);
  for (int i = 0; i < dofCount_; ++i) {
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

InlineSampleDofListProvider::InlineSampleDofListProvider()
{
  // Nothing to do
}

Teuchos::Array<int>
InlineSampleDofListProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  return getDofListFromParam(*params);
}

XMLFileSampleDofListProvider::XMLFileSampleDofListProvider()
{
  // Nothing to do
}

Teuchos::Array<int>
XMLFileSampleDofListProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string path = Teuchos::getParameter<std::string>(*params, "Sample Dof Input File Name");
  const Teuchos::RCP<const Teuchos::ParameterList> sampleDofsParams = Teuchos::getParametersFromXmlFile(path);
  return getDofListFromParam(*sampleDofsParams);
}

Teuchos::RCP<SampleDofListFactory> defaultSampleDofListFactoryNew(int dofCount) {
  const Teuchos::RCP<SampleDofListFactory> result(new SampleDofListFactory);

  result->extend("All", Teuchos::rcp(new AllSampleDofListProvider(dofCount)));
  result->extend("Inline", Teuchos::rcp(new InlineSampleDofListProvider));
  result->extend("File", Teuchos::rcp(new XMLFileSampleDofListProvider));

  return result;
}

} // end namespace Albany
