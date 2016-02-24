//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MORFacadeImpl.hpp"

#include "MOR_DefaultReducedBasisFactory.hpp"
#include "MOR_DefaultSampleDofListProviders.hpp"
#include "MOR_TruncatedReducedBasisSource.hpp"

#include "Albany_StkEpetraMVSource.hpp"
#include "Albany_DiscretizationDofListProvider.hpp"

namespace Albany {

MORFacadeImpl::MORFacadeImpl(
    const Teuchos::RCP<STKDiscretization> &disc,
    const Teuchos::RCP<Teuchos::ParameterList> &params) :
  basisFactory_(MOR::defaultReducedBasisFactoryNew(*disc->getMap())),
  samplingFactory_(MOR::defaultSampleDofListFactoryNew(disc->getMap())),
  spaceFactory_(new MOR::ReducedSpaceFactory(basisFactory_, samplingFactory_)),
  modelFactory_(new MOR::ReducedOrderModelFactory(spaceFactory_, params)),
  observerFactory_(new MOR::ObserverFactory(spaceFactory_, params))
{
  // Albany-specific reduced basis source
  const Teuchos::RCP<MOR::EpetraMVSource> stkMVSource(new StkEpetraMVSource(disc));
  basisFactory_->extend("Stk", Teuchos::rcp(new MOR::DefaultTruncatedReducedBasisSource(stkMVSource)));

  // Albany-specific sampling source
  samplingFactory_->extend("Stk", Teuchos::rcp(new DiscretizationSampleDofListProvider(disc)));
}

Teuchos::RCP<MOR::ReducedOrderModelFactory> MORFacadeImpl::modelFactory() const
{
  return modelFactory_;
}

Teuchos::RCP<MOR::ObserverFactory> MORFacadeImpl::observerFactory() const
{
  return observerFactory_;
}


Teuchos::RCP<MORFacade> createMORFacade(
    const Teuchos::RCP<AbstractDiscretization> &disc,
    const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const Teuchos::RCP<STKDiscretization> disc_actual =
    Teuchos::rcp_dynamic_cast<STKDiscretization>(disc);
  return Teuchos::rcp(new MORFacadeImpl(disc_actual, params));
}

} // end namespace Albany
