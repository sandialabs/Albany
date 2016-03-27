//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOR_DEFAULTSAMPLEDOFLISTPROVIDERS_HPP
#define MOR_DEFAULTSAMPLEDOFLISTPROVIDERS_HPP

#include "MOR_SampleDofListFactory.hpp"

#include "Teuchos_RCP.hpp"

class Epetra_BlockMap;

namespace MOR {

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

} // namespace MOR

#endif /* MOR_DEFAULTSAMPLEDOFLISTPROVIDERS_HPP */
