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

  int firstVectorRank = params->get("First Vector Rank", 0); // equiv... number of initial basis modes to skip
  int numModes = params->get("Basis Size Max", -1); // equiv... total number of basis modes
  int numFixedModes = params->get("Number of DBC Modes", 0);
  int numFreeModes = params->get("Number of Free Modes", -1);
  int lastVectorRank = std::max(numModes,numFixedModes+numFreeModes) + firstVectorRank;
  // We take the basis multivector to be everything from INDEX "firstVectorRank" (0-based counting), through (NOT INCLUDING) index "lastVectorRank".  In other words, we skip the first "firstVectorRank" modes and take the next "numModes" (or "numFixedModes+numFreeModes") ones.
  if (lastVectorRank>0) {
    if (firstVectorRank>0){
      return mvSource->truncatedMultiVectorNew(firstVectorRank, lastVectorRank);
    }
    else{
      return mvSource->truncatedMultiVectorNew(lastVectorRank);
    }
  } else {
    return mvSource->multiVectorNew();
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
