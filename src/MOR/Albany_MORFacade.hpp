//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef ALBANY_MORFACADE_HPP
#define ALBANY_MORFACADE_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace MOR {
class ReducedOrderModelFactory;
class ObserverFactory;
} // namespace MOR

namespace Albany {

class AbstractDiscretization;

class MORFacade {
public:
  virtual Teuchos::RCP<MOR::ReducedOrderModelFactory> modelFactory() const = 0;
  virtual Teuchos::RCP<MOR::ObserverFactory> observerFactory() const = 0;

  virtual ~MORFacade() {}
};

// Entry point defined in compilation unit of implementation
extern
Teuchos::RCP<MORFacade> createMORFacade(
    const Teuchos::RCP<AbstractDiscretization> &disc,
    const Teuchos::RCP<Teuchos::ParameterList> &params);

} // end namespace Albany

#endif /* ALBANY_MORFACADE_HPP */
