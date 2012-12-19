//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SAMPLEDOFLISTFACTORY_HPP
#define ALBANY_SAMPLEDOFLISTFACTORY_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

#include <string>
#include <map>

namespace Albany {

class SampleDofListFactory {
public:
  SampleDofListFactory();

  Teuchos::Array<int> create(const Teuchos::RCP<Teuchos::ParameterList> &params);

  class DofListProvider;
  void extend(const std::string &id, const Teuchos::RCP<DofListProvider> &provider);

private:
  typedef std::map<std::string, Teuchos::RCP<DofListProvider> > ProviderMap;
  ProviderMap providers_;
};

class SampleDofListFactory::DofListProvider {
public:
  virtual Teuchos::Array<int> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params) = 0;
  virtual ~DofListProvider() {}
};

} // end namespace Albany

#endif /* ALBANY_SAMPLEDOFLISTFACTORY_HPP */
