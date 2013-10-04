//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MORFacade.hpp"

#include "MOR_ReducedBasisFactory.hpp"
#include "MOR_ReducedSpaceFactory.hpp"
#include "MOR_SampleDofListFactory.hpp"
#include "MOR_DefaultSampleDofListProviders.hpp"

#include "MOR_ReducedOrderModelFactory.hpp"
#include "MOR_ObserverFactory.hpp"

#include "MOR_IdentityBasisSource.hpp"
#include "MOR_BasisInputFile.hpp"

#include "Albany_StkBasisProvider.hpp"
#include "Albany_DiscretizationDofListProvider.hpp"

#include "Albany_STKDiscretization.hpp"

namespace Albany {

class MORFacadeImpl : public MORFacade {
public:
  MORFacadeImpl(
      const Teuchos::RCP<STKDiscretization> &disc,
      const Teuchos::RCP<Teuchos::ParameterList> &params);

  virtual Teuchos::RCP<MOR::ReducedOrderModelFactory> modelFactory() const;
  virtual Teuchos::RCP<MOR::ObserverFactory> observerFactory() const;

private:
  Teuchos::RCP<MOR::ReducedBasisFactory> basisFactory_;
  Teuchos::RCP<MOR::SampleDofListFactory> samplingFactory_;
  Teuchos::RCP<MOR::ReducedSpaceFactory> spaceFactory_;

  Teuchos::RCP<MOR::ReducedOrderModelFactory> modelFactory_;
  Teuchos::RCP<MOR::ObserverFactory> observerFactory_;
};

Teuchos::RCP<MOR::ReducedOrderModelFactory> MORFacadeImpl::modelFactory() const
{
  return modelFactory_;
}

Teuchos::RCP<MOR::ObserverFactory> MORFacadeImpl::observerFactory() const
{
  return observerFactory_;
}

MORFacadeImpl::MORFacadeImpl(
    const Teuchos::RCP<STKDiscretization> &disc,
    const Teuchos::RCP<Teuchos::ParameterList> &params) :
  basisFactory_(new MOR::ReducedBasisFactory),
  samplingFactory_(MOR::defaultSampleDofListFactoryNew(disc->getMap())),
  spaceFactory_(new MOR::ReducedSpaceFactory(basisFactory_, samplingFactory_)),
  modelFactory_(new MOR::ReducedOrderModelFactory(spaceFactory_, params)),
  observerFactory_(new MOR::ObserverFactory(spaceFactory_, params))
{
  basisFactory_->extend("Identity", Teuchos::rcp(new MOR::IdentityBasisSource(*disc->getMap())));
  basisFactory_->extend("File", Teuchos::rcp(new MOR::BasisInputFile(*disc->getMap())));
  basisFactory_->extend("Stk", Teuchos::rcp(new StkBasisProvider(disc)));

  samplingFactory_->extend("Stk", Teuchos::rcp(new DiscretizationSampleDofListProvider(disc)));
}

Teuchos::RCP<MORFacade> createMORFacade(
    const Teuchos::RCP<AbstractDiscretization> &disc,
    const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const Teuchos::RCP<STKDiscretization> disc_actual =
    Teuchos::rcp_dynamic_cast<STKDiscretization>(disc);
  return Teuchos::rcp(new MORFacadeImpl(disc_actual, params));
}

}
