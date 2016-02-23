//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_TRUNCATEDUCEDBASISSOURCE_HPP
#define MOR_TRUNCATEDUCEDBASISSOURCE_HPP

#include "MOR_ReducedBasisSource.hpp"

#include "MOR_EpetraMVSource.hpp"

#include "Teuchos_Ptr.hpp"

namespace MOR {

template <typename EpetraMVSourceProvider>
class TruncatedReducedBasisSource : public ReducedBasisSource {
public:
  explicit TruncatedReducedBasisSource(const EpetraMVSourceProvider &provider) :
    provider_(provider)
  {}

  virtual ReducedBasisElements operator()(const Teuchos::RCP<Teuchos::ParameterList> &params);

private:
  EpetraMVSourceProvider provider_;
};

template <typename EpetraMVSourceProvider>
ReducedBasisElements
TruncatedReducedBasisSource<EpetraMVSourceProvider>::operator()(
    const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const Teuchos::RCP<BasicEpetraMVSource> mvSource = provider_(params);

  const Teuchos::Ptr<const int> vectorCountMax(params->getPtr<int>("Basis Size Max"));
  if (Teuchos::nonnull(vectorCountMax)) {
    return mvSource->truncatedMultiVectorNew(*vectorCountMax);
  } else {
    return mvSource->multiVectorNew();;
  }
}


class EpetraMVSourceInstanceProvider {
public:
  /*implicit*/ EpetraMVSourceInstanceProvider(
      const Teuchos::RCP<BasicEpetraMVSource> &instance) :
    instance_(instance)
  {}

  Teuchos::RCP<BasicEpetraMVSource> operator()(
      const Teuchos::RCP<Teuchos::ParameterList> &/*params*/)
  {
    return instance_;
  }

private:
  Teuchos::RCP<BasicEpetraMVSource> instance_;
};


class DefaultTruncatedReducedBasisSource :
  public TruncatedReducedBasisSource<EpetraMVSourceInstanceProvider> {
public:
  explicit DefaultTruncatedReducedBasisSource(
      const Teuchos::RCP<BasicEpetraMVSource> &instance) :
    TruncatedReducedBasisSource<EpetraMVSourceInstanceProvider>(instance)
  {}
};

} // end namepsace Albany

#endif /* MOR_TRUNCATEDUCEDBASISSOURCE_HPP */
