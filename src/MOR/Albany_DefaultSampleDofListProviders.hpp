//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DEFAULTSAMPLEDOFLISTPROVIDERS_HPP
#define ALBANY_DEFAULTSAMPLEDOFLISTPROVIDERS_HPP

#include "Albany_SampleDofListFactory.hpp"

#include "Teuchos_RCP.hpp"

class Epetra_BlockMap;

namespace Albany {

class AllSampleDofListProvider : public SampleDofListFactory::DofListProvider {
public:
  explicit AllSampleDofListProvider(const Teuchos::RCP<const Epetra_BlockMap> &map);
  virtual Teuchos::Array<int> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Teuchos::RCP<const Epetra_BlockMap> map_;
};

class InlineSampleDofListProvider : public SampleDofListFactory::DofListProvider {
public:
  explicit InlineSampleDofListProvider(const Teuchos::RCP<const Epetra_BlockMap> &map);
  virtual Teuchos::Array<int> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Teuchos::RCP<const Epetra_BlockMap> map_;
};

class XMLFileSampleDofListProvider : public SampleDofListFactory::DofListProvider {
public:
  explicit XMLFileSampleDofListProvider(const Teuchos::RCP<const Epetra_BlockMap> &map);
  virtual Teuchos::Array<int> operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  Teuchos::RCP<const Epetra_BlockMap> map_;
};

Teuchos::RCP<SampleDofListFactory> defaultSampleDofListFactoryNew(const Teuchos::RCP<const Epetra_BlockMap> &map);

} // end namespace Albany

#endif /* ALBANY_DEFAULTSAMPLEDOFLISTPROVIDERS_HPP */
