//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_REDUCEDBASISFACTORY_HPP
#define MOR_REDUCEDBASISFACTORY_HPP

#include "MOR_ReducedBasisElements.hpp"
#include "MOR_ReducedBasisSource.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <string>
#include <map>

namespace MOR {

class ReducedBasisFactory {
public:
  ReducedBasisFactory();

  ReducedBasisElements create(const Teuchos::RCP<Teuchos::ParameterList> &params);

  void extend(const std::string &id, const Teuchos::RCP<ReducedBasisSource> &source);

private:
  typedef std::map<std::string, Teuchos::RCP<ReducedBasisSource> > BasisSourceMap;
  BasisSourceMap sources_;
};

} // end namespace MOR

#endif /* MOR_REDUCEDBASISFACTORY_HPP */
