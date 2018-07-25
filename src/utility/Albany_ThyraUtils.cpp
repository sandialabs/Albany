#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_Utils.hpp"

#include "Thyra_VectorStdOps.hpp"

namespace Albany
{

// ========= Thyra_LinearOp utilities ========= //

bool isFillActive (const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tlop.is_null()) {
    return tmat->isFillActive();
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return false;
}

void resumeFill (const Teuchos::RCP<Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tlop.is_null()) {
    tmat->resumeFill();
    return;
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void fillComplete (const Teuchos::RCP<Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tlop.is_null()) {
    tmat->fillComplete();
    return;
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void assign (const Teuchos::RCP<Thyra_LinearOp>& lop, const ST value)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tlop.is_null()) {
    tmat->setAllToScalar(value);
    return;
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void getDiagonalCopy (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                      Teuchos::RCP<Thyra_Vector>& diag)
{
  // Diagonal makes sense only for (globally) square operators.
  // From Thyra, we can't check the global ids of the range/domain vector spaces,
  // but at least we can check that they have the same (global) dimension.
  TEUCHOST_TEST_FOR_EXCEPTION(lop->range()->dim()!=lop->domain()->dim(), std::logic_error,
                              "Error! Attempt to take the diagonal of a non-square operator.\n");

  // If diag is not created, do it.
  if (diag.is_null()) {
    diag = Thyra::createMember(lop->range());
  }

  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tlop.is_null()) {
    tmat->getLocalDiagCopy(*Albany::getTpetraVector(diag,true));
    return;
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

// ========= Thyra_Vector utilities ========== //

Teuchos::ArrayRCP<ST> getNonconstLocalData (const Teuchos::RCP<Thyra_Vector>& v)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getTpetraVector(v,false);
  if (!tv.is_null()) {
    auto view = tv->getLocalView<Kokkos::HostSpace>();
    tv->modify<Kokkos::HostSpace>();
    Teuchos::ArrayRCP<impl_scalar_type> dataAsArcp =
        Kokkos::Compat::persistingView (X_lcl);
    return Teuchos::arcp_reinterpret_cast<Scalar> (dataAsArcp);
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  Teuchos::ArrayRCP<ST> dummy;
  return dummy;
}

Teuchos::ArrayRCP<const ST> getLocalData (const Teuchos::RCP<const Thyra_Vector>& v)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getConstTpetraVector(v,false);
  if (!tv.is_null()) {
    // Here we can use the get1dView method of Tpetra::MultiVector, since we do not need
    // (and should not) mark the data as modified.
    return tv->get1dView();
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  Teuchos::ArrayRCP<const ST> dummy;
  return dummy;
}

void scale_and_update (const Teuchos::RCP<Thyra_Vector> y, const ST y_coeff,
                       const Teuchos::RCP<const Thyra_Vector> x, const ST x_coeff)
{
  Thyra::V_StVpStV(y.ptr(),x_coeff,*x,y_coeff,*y);
}

// ========= Matrix Market utilities ========== //

void
writeMatrixMarket(
    const Teuchos::RCP<const Thyra_Vector>& v, std::string const& prefix,
    int const counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getConstTpetraVector(v,false);
  if (!tv.is_null()) {
    writeMatrixMarket(tv,prefix,counter);
    return;
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
}

void
writeMatrixMarket(
    const Teuchos::Array<Teuchos::RCP<const Thyra_Vector>>& vs,
    const std::string& prefix, int const counter = -1)
{
  Teuchos::Array<Teuchos::RCP<const Tpetra_Vector>> tvs(vs.size());

  bool first_vector = true;
  bool successful_cast = false;
  for (int i=0; i<vs.size(); ++i) {
    // Allow failure at first vector, since we don't know what the underlying linear algebra is
    // If the first one goes through, however, we want the next ones to go through as well (no mixed types!)
    tvs[i] = getConstTpetraVector(vs[i],!first_vector);
    if (tvs[i].is_null()) {
      TEUCHOS_TEST_FOR_EXCEPTION(!first_vector, std::logic_error, "Error! I was able to cast one Thyra_Vector to Tpetra_Vector, but not the rest.\n");
      break;
    }
    first_vector = false;
    successful_cast = true;
  }
  if (successful_cast) {
    writeMatrixMarket(tvs,prefix,counter);
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (!successful_cast, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
}

void
writeMatrixMarket(
    const Teuchos::RCP<const Thyra_LinearOp>& A, std::string const& prefix,
    int const counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tA = getConstTpetraMatrix(A,false);
  if (!tv.is_null()) {
    writeMatrixMarket(tA,prefix,counter);
    return;
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void
writeMatrixMarket(
    const Teuchos::Array<Teuchos::RCP<const Thyra_LinearOp>>& As,
    const std::string& prefix, int const counter = -1)
{
  Teuchos::Array<Teuchos::RCP<const Tpetra_CrsMatrix>> tAs(As.size());

  bool first_op = true;
  bool successful_cast = false;
  for (int i=0; i<As.size(); ++i) {
    // Allow failure at first operator, since we don't know what the underlying linear algebra is
    // If the first one goes through, however, we want the next ones to go through as well (no mixed types!)
    tAs[i] = getConstTpetraMatrix(As[i],!first_op);
    if (tAs[i].is_null()) {
      TEUCHOS_TEST_FOR_EXCEPTION(!first_op, std::logic_error, "Error! I was able to cast one Thyra_LinearOp to Tpetra_CrsMatrix, but not the rest.\n");
      break;
    }
    first_op = false;
    successful_cast = true;
  }
  if (successful_cast) {
    writeMatrixMarket(tAs,prefix,counter);
  }

  // If all the tries above are not successful, throw an error.
  TEUCHOST_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

// ========= Thyra_ProductXYZ utilities ========== //

Teuchos::RCP<const Thyra_ProductVectorSpace>
getProductVectorSpace (const Teuchos::RCP<const Thyra_VectorSpace> vs,
                       const bool throw_on_failure)
{
  Teuchos::RCP<const Thyra_ProductVectorSpace> pvs;
  pvs = Teuchos::rcp_dynamic_cast<const Thyra_ProductVectorSpace>(vs,throw_on_failure)
  return pvs;
}

Teuchos::RCP<Thyra_ProductVector>
getProductVector (const Teuchos::RCP<Thyra_Vector> v,
                  const bool throw_on_failure)
{
  Teuchos::RCP<Thyra_ProductVector> pv;
  pv = Teuchos::rcp_dynamic_cast<Thyra_ProductVector>(v,throw_on_failure);
  return sv;
}

Teuchos::RCP<const Thyra_ProductVector>
getConstProductVector (const Teuchos::RCP<const Thyra_Vector> v,
                       const bool throw_on_failure)
{
  Teuchos::RCP<const Thyra_ProductVector> pv;
  pv = Teuchos::rcp_dynamic_cast<const Thyra_ProductVector>(v,throw_on_failure);
  return pv;
}

Teuchos::RCP<Thyra_ProductMultiVector>
getProductMultiVector (const Teuchos::RCP<Thyra_MultiVector> mv,
                       const bool throw_on_failure)
{
  Teuchos::RCP<Thyra_ProductMultiVector> pmv;
  pmv = Teuchos::rcp_dynamic_cast<Thyra_ProductMultiVector>(mv,throw_on_failure);
  return pvs;
}

Teuchos::RCP<const Thyra_ProductMultiVector>
getConstProductMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv,
                            const bool throw_on_failure)
{
  Teuchos::RCP<const Thyra_ProductMultiVector> pmv;
  pmv = Teuchos::rcp_dynamic_cast<const Thyra_ProductMultiVector>(mv,throw_on_failure);
  return pvs;
}

} // namespace Albany
