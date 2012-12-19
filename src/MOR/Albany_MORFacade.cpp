//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MORFacade.hpp"

#include "Albany_LinearReducedSpaceFactory.hpp"
#include "Albany_SampleDofListFactory.hpp"
#include "Albany_DefaultSampleDofListProviders.hpp"

#include "Albany_ReducedOrderModelFactory.hpp"
#include "Albany_MORObserverFactory.hpp"

#include "Albany_BasisInputFile.hpp"
#include "Albany_StkBasisProvider.hpp"
#include "Albany_DiscretizationDofListProvider.hpp"

#include "Albany_STKDiscretization.hpp"

namespace Albany {

class MORFacadeImpl : public MORFacade {
public:
  MORFacadeImpl(
      const Teuchos::RCP<STKDiscretization> &disc,
      const Teuchos::RCP<Teuchos::ParameterList> &params);

  virtual Teuchos::RCP<ReducedOrderModelFactory> modelFactory() const;
  virtual Teuchos::RCP<MORObserverFactory> observerFactory() const;

private:
  Teuchos::RCP<LinearReducedSpaceFactory> spaceFactory_;
  Teuchos::RCP<SampleDofListFactory> samplingFactory_;

  Teuchos::RCP<ReducedOrderModelFactory> modelFactory_;
  Teuchos::RCP<MORObserverFactory> observerFactory_;
};

Teuchos::RCP<ReducedOrderModelFactory> MORFacadeImpl::modelFactory() const
{
  return modelFactory_;
}

Teuchos::RCP<MORObserverFactory> MORFacadeImpl::observerFactory() const
{
  return observerFactory_;
}

MORFacadeImpl::MORFacadeImpl(
    const Teuchos::RCP<STKDiscretization> &disc,
    const Teuchos::RCP<Teuchos::ParameterList> &params) :
  spaceFactory_(new LinearReducedSpaceFactory),
  samplingFactory_(defaultSampleDofListFactoryNew(disc->getMap())),
  modelFactory_(new ReducedOrderModelFactory(spaceFactory_, samplingFactory_, params)),
  observerFactory_(new MORObserverFactory(spaceFactory_, params))
{
  spaceFactory_->extend("File", Teuchos::rcp(new BasisInputFile(*disc->getMap())));
  spaceFactory_->extend("Stk", Teuchos::rcp(new StkBasisProvider(disc)));

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
