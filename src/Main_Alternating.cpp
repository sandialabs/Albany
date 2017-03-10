//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#if defined(ALBANY_CHECK_FPE)
#include <math.h>
#include <xmmintrin.h>
#endif

#if defined(ALBANY_FLUSH_DENORMALS)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#include "Albany_DataTypes.hpp"
#include "Albany_Memory.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_Utils.hpp"
#include "Kokkos_Core.hpp"
#include "Phalanx_config.hpp"
#include "Phalanx.hpp"
#include "Piro_PerformSolve.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

// Global variable that denotes this is a Tpetra executable
// amota: Why is this needed at all?
bool
TpetraBuild{true};

using ThyraResponses =
  Teuchos::Array<Teuchos::RCP<Thyra::VectorBase<ST> const>>;

using ThyraSensitivities =
  Teuchos::Array<Teuchos::Array<Teuchos::RCP<Thyra::MultiVectorBase<ST> const>>>;

using Responses =
    Teuchos::Array<Teuchos::RCP<Tpetra_Vector const>>;

using Sensitivities =
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<Tpetra_MultiVector const>>>;

namespace {

void
tpetraFromThyra(
    ThyraResponses const & thyra_responses,
    ThyraSensitivities const & thyra_sensitivities,
    Responses & responses,
    Sensitivities & sensitivities)
{
  responses.clear();
  responses.reserve(thyra_responses.size());

  for (auto && rcp_vb : thyra_responses) {
    if (Teuchos::nonnull(rcp_vb) == true) {
      responses.push_back(ConverterT::getConstTpetraVector(rcp_vb));
    } else {
      responses.push_back(Teuchos::null);
    }
  }

  sensitivities.clear();
  sensitivities.reserve(thyra_sensitivities.size());

  for (auto && arcp_mvb : thyra_sensitivities) {
    Teuchos::Array<Teuchos::RCP<const Tpetra_MultiVector>>
    sensitivity;

    sensitivity.reserve(arcp_mvb.size());

    for (auto && rcp_mvb : arcp_mvb) {
      if (Teuchos::nonnull(rcp_mvb) == true) {
        sensitivity.push_back(ConverterT::getConstTpetraMultiVector(rcp_mvb));
      } else {
        sensitivity.push_back(Teuchos::null);
      }
    }
    sensitivities.push_back(sensitivity);
  }

  return;
}

} // anonymous namespace

int main(int ac, char *av[])
{
#if defined(ALBANY_FLUSH_DENORMALS)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

#if defined(ALBANY_CHECK_FPE)
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif

#if defined(ALBANY_64BIT_INT)
  ALBANY_ASSERT(sizeof(long) == 8, "64-bit Albany requires sizeof(long) == 8");
#endif

  // 0 = pass, failures are incremented
  int
  status{0};

  bool
  success{true};

  auto &&
  fos{*Teuchos::VerboseObjectBase::getDefaultOStream()};

  fos << "Schwarz alternating method" << std::endl;

  return status;
}
