//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_MORFACADEIMPL_HPP
#define ALBANY_MORFACADEIMPL_HPP

#include "Albany_MORFacade.hpp"

#include "MOR_ReducedBasisFactory.hpp"
#include "MOR_ReducedSpaceFactory.hpp"
#include "MOR_SampleDofListFactory.hpp"

#include "MOR_ReducedOrderModelFactory.hpp"
#include "MOR_ObserverFactory.hpp"

#include "Albany_STKDiscretization.hpp"

namespace Albany {

class MORFacadeImpl : public MORFacade {
public:
  MORFacadeImpl(
      const Teuchos::RCP<STKDiscretization> &disc,
      const Teuchos::RCP<Teuchos::ParameterList> &params);

  virtual Teuchos::RCP<MOR::ReducedOrderModelFactory> modelFactory() const;
  virtual Teuchos::RCP<MOR::ObserverFactory> observerFactory() const;

  MOR::ReducedBasisFactory &basisFactory();
  MOR::SampleDofListFactory &samplingFactory();
  MOR::ReducedSpaceFactory &spaceFactory();

private:
  Teuchos::RCP<MOR::ReducedBasisFactory> basisFactory_;
  Teuchos::RCP<MOR::SampleDofListFactory> samplingFactory_;
  Teuchos::RCP<MOR::ReducedSpaceFactory> spaceFactory_;

  Teuchos::RCP<MOR::ReducedOrderModelFactory> modelFactory_;
  Teuchos::RCP<MOR::ObserverFactory> observerFactory_;
};

inline
MOR::ReducedBasisFactory &
MORFacadeImpl::basisFactory()
{
  return *basisFactory_;
}

inline
MOR::SampleDofListFactory &
MORFacadeImpl::samplingFactory()
{
  return *samplingFactory_;
}

inline
MOR::ReducedSpaceFactory &
MORFacadeImpl::spaceFactory()
{
  return *spaceFactory_;
}


// Entry point also declared with base class Albany::MORFacade
Teuchos::RCP<MORFacade> createMORFacade(
    const Teuchos::RCP<AbstractDiscretization> &disc,
    const Teuchos::RCP<Teuchos::ParameterList> &params);

} // end namespace Albany

#endif /* ALBANY_MORFACADEIMPL_HPP */
