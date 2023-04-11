#include "Albany_SacadoTypes.hpp"

#include "Teuchos_UnitTestHelpers.hpp"
#include "Teuchos_LocalTestingHelpers.hpp"
#include "Sacado_Fad_GeneralFadTestingHelpers.hpp"

TEUCHOS_UNIT_TEST(FadTypes, HessianFad)
{
  using scalarType = typename Sacado::ScalarType<HessianVecFad>::type;
  const scalarType tol = 1000.0 * std::numeric_limits<scalarType>::epsilon();

  HessianVecInnerFad tmp (0);

  // Use max to keep static-length for SFad
  HessianVecFad a,b;
  a = HessianVecFad(std::max(a.size(),1),tmp);
  b = HessianVecFad(std::max(b.size(),1),tmp);

  a.val().val() = 2.;
  b.val().val() = 2.;

  a.val().fastAccessDx(0) = 2.;
  b.val().fastAccessDx(0) = 2.;

  a.fastAccessDx(0).fastAccessDx(0) = 1.;
  b.fastAccessDx(0).fastAccessDx(0) = 1.;

  TEST_FLOATING_EQUALITY(a,b,tol);

  b.fastAccessDx(0).fastAccessDx(0) = 2.;

  TEUCHOS_TEST_FLOATING_NOT_EQUALITY(a,b,tol,out,success);
}
