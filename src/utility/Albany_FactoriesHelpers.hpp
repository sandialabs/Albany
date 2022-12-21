#ifndef ALBANY_FACTORIES_HELPERS_HPP
#define ALBANY_FACTORIES_HELPERS_HPP

// WARNING: DO NOT include this file in albanyLib.
//          This is a header-only utility for libs/executables that need 
//          the albany factories correctly setup before all the automatic
//          constructions are invoked

#include "Albany_config.h"
#include "Albany_ProblemFactory.hpp"
#include "Albany_CoreProblemFactory.hpp"
#ifdef ALBANY_DEMO_PDES
#include "Albany_DemoProblemFactory.hpp"
#endif
#ifdef ALBANY_LANDICE
#include "LandIce_ProblemFactory.hpp"
#endif

namespace Albany {

void register_pb_factories () {
  auto& pb_factories = FactoriesContainer<ProblemFactory>::instance();

  pb_factories.add_factory(CoreProblemFactory::instance());
#ifdef ALBANY_DEMO_PDES
  pb_factories.add_factory(DemoProblemFactory::instance());
#endif
#ifdef ALBANY_LANDICE
  pb_factories.add_factory(LandIce::LandIceProblemFactory::instance());
#endif
}

} // namespace Albany

#endif // ALBANY_FACTORIES_HELPERS_HPP
