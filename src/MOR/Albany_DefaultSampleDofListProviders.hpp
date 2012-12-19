//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DEFAULTSAMPLEDOFLISTPROVIDERS_HPP
#define ALBANY_DEFAULTSAMPLEDOFLISTPROVIDERS_HPP

#include "Albany_SampleDofListFactory.hpp"

namespace Albany {

class AllSampleDofListProvider : public SampleDofListFactory::DofListProvider {
public:
  explicit AllSampleDofListProvider(int dofCount);
  virtual Teuchos::Array<int> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  int dofCount_;
};

class InlineSampleDofListProvider : public SampleDofListFactory::DofListProvider {
public:
  InlineSampleDofListProvider();
  virtual Teuchos::Array<int> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);
};

class XMLFileSampleDofListProvider : public SampleDofListFactory::DofListProvider {
public:
  XMLFileSampleDofListProvider();
  virtual Teuchos::Array<int> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);
};

Teuchos::RCP<SampleDofListFactory> defaultSampleDofListFactoryNew(int dofCount);

} // end namespace Albany

#endif /* ALBANY_DEFAULTSAMPLEDOFLISTPROVIDERS_HPP */
