#include "Albany_Application.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms)
{
  return app->getNumEquations() * ms->ctd.node_count;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Tangent> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms)
{
  //amb Need to figure this out. Unlike the Jacobian case, it appears that in
  // the Tangent case, it's OK to overestimate the size.
  return 32;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms)
{
  //amb Need to figure out.
  return getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(app, ms);
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian> (
 const Albany::Application* app, const int ebi)
{
  return getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
    app, app->getDiscretization()->getMeshStruct()->getMeshSpecs()[ebi].get());
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Tangent> (
 const Albany::Application* app, const int ebi)
{
  return getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(
    app, app->getDiscretization()->getMeshStruct()->getMeshSpecs()[ebi].get());
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv> (
 const Albany::Application* app, const int ebi)
{
  return getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
    app, app->getDiscretization()->getMeshStruct()->getMeshSpecs()[ebi].get());
}

} // namespace PHAL
