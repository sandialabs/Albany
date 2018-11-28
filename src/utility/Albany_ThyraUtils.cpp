#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_Utils.hpp"

#include "Petra_Converters.hpp"

#include "Thyra_VectorStdOps.hpp"

#if defined(ALBANY_EPETRA)
#include "AztecOO_ConditionNumber.h"
#endif

namespace Albany
{

// ========= Thyra_LinearOp utilities ========= //

bool isFillActive (const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    return tmat->isFillActive();
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return false;
}

void resumeFill (const Teuchos::RCP<Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    tmat->resumeFill();
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void fillComplete (const Teuchos::RCP<Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    tmat->fillComplete();
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void assign (const Teuchos::RCP<Thyra_LinearOp>& lop, const ST value)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    // Tpetra throws when trying to set scalars in an already filled matrix
    bool callFillComplete = false;
    if(!tmat->isFillActive()){
       tmat->resumeFill();
       callFillComplete = true;
    }

    tmat->setAllToScalar(value);

    if(callFillComplete){
      tmat->fillComplete();
    }

    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void getDiagonalCopy (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                      Teuchos::RCP<Thyra_Vector>& diag)
{
  // Diagonal makes sense only for (globally) square operators.
  // From Thyra, we can't check the global ids of the range/domain vector spaces,
  // but at least we can check that they have the same (global) dimension.
  TEUCHOS_TEST_FOR_EXCEPTION(lop->range()->dim()!=lop->domain()->dim(), std::logic_error,
                              "Error! Attempt to take the diagonal of a non-square operator.\n");

  // If diag is not created, do it.
  if (diag.is_null()) {
    diag = Thyra::createMember(lop->range());
  }

  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    tmat->getLocalDiagCopy(*Albany::getTpetraVector(diag,true));
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void getLocalRowValues (const Teuchos::RCP<Thyra_LinearOp>& lop,
                        const LO lrow,
                        Teuchos::Array<LO>& indices,
                        Teuchos::Array<ST>& values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    auto numEntries = tmat->getNumEntriesInLocalRow(lrow);
    indices.resize(numEntries);
    values.resize(numEntries);
    tmat->getLocalRowCopy(lrow,indices,values,numEntries);
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void addToLocalRowValues (const Teuchos::RCP<Thyra_LinearOp>& lop,
                          const LO lrow,
                          const Teuchos::ArrayView<const LO> indices,
                          const Teuchos::ArrayView<const ST> values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    tmat->sumIntoLocalValues(lrow,indices,values);
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void setLocalRowValues (const Teuchos::RCP<Thyra_LinearOp>& lop,
                        const LO lrow,
                        const Teuchos::ArrayView<const LO> indices,
                        const Teuchos::ArrayView<const ST> values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    tmat->replaceLocalValues(lrow,indices,values);
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

double computeConditionNumber (const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
#ifdef ALBANY_EPETRA
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    Petra::Converter converter(tmat->getComm());

    auto emat = Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(tmat,converter.commE_);
    AztecOOConditionNumber conditionEstimator;
    conditionEstimator.initialize(*emat);
    int maxIters = 40000;
    double tol = 1e-10;
    int status = conditionEstimator.computeConditionNumber(maxIters, tol);
    if (status != 0) {
      auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
      *out << "WARNING: AztecOO::ConditionNumber::computeConditionNumber returned "
           << "non-zero status = " << status
           << ".  Condition number estimate may be wrong!\n";
    }
    double condest = conditionEstimator.getConditionNumber();
    return condest;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
#else
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Condition number estimation requires ALBANY_EPETRA.\n");
  // Suppress compiler warning for unused argument
  (void) lop;
#endif

  // Dummy return value to silence compiler warning
  return 0.0;
}

DeviceLocalMatrix<const ST> getDeviceData (const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    // Get the local matrix from tpetra.
    DeviceLocalMatrix<const ST> data = tmat->getLocalMatrix();
    return data;
  }

  // TODO: to add epetra, create device matrix, get values view, create host copy, extract data
  //       from matrix into values host view, deep copy to device. We could otherwise ENFORCE
  //       PHX::Device to be a host exec space for Epetra.

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceLocalMatrix<const ST> dummy;
  return dummy;
}

DeviceLocalMatrix<ST> getNonconstDeviceData (const Teuchos::RCP<Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    // Get the local matrix from tpetra.
    DeviceLocalMatrix<ST> data = tmat->getLocalMatrix();
    return data;
  }

  // TODO: to add epetra, create device matrix, get values view, create host copy, extract data
  //       from matrix into values host view, deep copy to device. We could otherwise ENFORCE
  //       PHX::Device to be a host exec space for Epetra.

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceLocalMatrix<ST> dummy;
  return dummy;
}

// ========= Thyra_Vector utilities ========== //

// TODO: probably we should do something like
//
//  spmd_v = cast_to_spmd_vector(v);
//  ArrayRCP<ST> a;
//  spmd_v->getLocalData(Teuchos::outarg(a));
//  return a;
//
// This would work for ALL concrete impl of Thyra_Vector,
// provided that they inherit from SpmdVectorBase, which
// is a reasonable assumption (hold for both Tpetra and Epetra)

Teuchos::ArrayRCP<ST> getNonconstLocalData (const Teuchos::RCP<Thyra_Vector>& v)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getTpetraVector(v,false);
  if (!tv.is_null()) {
    // (and should not) mark the data as modified.
    return tv->get1dViewNonConst();
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

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

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  Teuchos::ArrayRCP<const ST> dummy;
  return dummy;
}

Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>
getNonconstLocalData (const Teuchos::RCP<Thyra_MultiVector>& mv)
{
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> data(mv->domain()->dim());
  for (int i=0; i<mv->domain()->dim(); ++i) {
    data[i] = getNonconstLocalData(mv->col(i));
  }
  return data;
}

Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>>
getLocalData (const Teuchos::RCP<const Thyra_MultiVector>& mv)
{
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> data(mv->domain()->dim());
  for (int i=0; i<mv->domain()->dim(); ++i) {
    data[i] = getLocalData(mv->col(i));
  }
  return data;
}

DeviceView1d<const ST> getDeviceData (const Teuchos::RCP<const Thyra_Vector>& v)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getConstTpetraVector(v,false);
  if (!tv.is_null()) {
    auto data2d = tv->getLocalView<KokkosNode::execution_space>();
    DeviceView1d<const ST> data = Kokkos::subview(data2d, Kokkos::ALL(), 0);
    return data;
  }

  // TODO: to add epetra, create device view, create host copy, extract data
  //       into host view, deep copy to device. We could otherwise ENFORCE
  //       PHX::Device to be a host exec space for Epetra.

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceView1d<const ST> dummy;
  return dummy;
}

DeviceView1d<ST> getNonconstDeviceData (const Teuchos::RCP<Thyra_Vector>& v)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getTpetraVector(v,false);
  if (!tv.is_null()) {
    auto data2d = tv->getLocalView<KokkosNode::execution_space>();
    DeviceView1d<ST> data = Kokkos::subview(data2d, Kokkos::ALL(), 0);
    return data;
  }

  // TODO: to add epetra, create device view, create host copy, extract data
  //       into host view, deep copy to device. We could otherwise ENFORCE
  //       PHX::Device to be a host exec space for Epetra.

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceView1d<ST> dummy;
  return dummy;
}

void scale_and_update (const Teuchos::RCP<Thyra_Vector> y, const ST y_coeff,
                       const Teuchos::RCP<const Thyra_Vector> x, const ST x_coeff)
{
  Thyra::V_StVpStV(y.ptr(),x_coeff,*x,y_coeff,*y);
}

// ======== Object printing utilities ========= //

template<>
void describe<Thyra_Vector> (const Teuchos::RCP<const Thyra_Vector>& v,
                             Teuchos::FancyOStream& out,
                             const Teuchos::EVerbosityLevel verbLevel)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getConstTpetraVector(v,false);
  if (!tv.is_null()) {
    tv->describe(out,verbLevel);
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
}

template<>
void describe<Thyra_LinearOp> (const Teuchos::RCP<const Thyra_LinearOp>& op,
                               Teuchos::FancyOStream& out,
                               const Teuchos::EVerbosityLevel verbLevel)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto top = getConstTpetraOperator(op,false);
  if (!top.is_null()) {
    top->describe(out,verbLevel);
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
}

// ========= Matrix Market utilities ========== //

// These routines implement a specialization of the template functions declared in Albany_Utils.hpp
template<>
void
writeMatrixMarket<Thyra_MultiVector>(
    const Teuchos::RCP<const Thyra_MultiVector>& mv,
    const std::string& prefix,
    const int counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmv = getConstTpetraMultiVector(mv,false);
  if (!tmv.is_null()) {
    writeMatrixMarket(tmv,prefix,counter);
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
}

template<>
void
writeMatrixMarket<Thyra_LinearOp>(
    const Teuchos::RCP<const Thyra_LinearOp>& A,
    const std::string& prefix,
    const int counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tA = getConstTpetraMatrix(A,false);
  if (!tA.is_null()) {
    writeMatrixMarket(tA,prefix,counter);
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

// These routines implement a specialization of the template functions declared in Albany_Utils.hpp
template<>
void
writeMatrixMarket<Thyra_VectorSpace>(
    const Teuchos::RCP<const Thyra_VectorSpace>& vs,
    const std::string& prefix,
    const int counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tm = getTpetraMap(vs,false);
  if (!tm.is_null()) {
    writeMatrixMarket(tm,prefix,counter);
    return;
  }

  // TODO: add epetra

  // If all the tries above are not successful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");
}

// ========= Thyra_SpmdXYZ utilities ========== //

Teuchos::RCP<const Thyra_SpmdVectorSpace>
getSpmdVectorSpace (const Teuchos::RCP<const Thyra_VectorSpace> vs,
                    const bool throw_on_failure)
{
  Teuchos::RCP<const Thyra_SpmdVectorSpace> spmd_vs;
  spmd_vs = Teuchos::rcp_dynamic_cast<const Thyra_SpmdVectorSpace>(vs,throw_on_failure);
  return spmd_vs;
}

// ========= Thyra_ProductXYZ utilities ========== //

Teuchos::RCP<const Thyra_ProductVectorSpace>
getProductVectorSpace (const Teuchos::RCP<const Thyra_VectorSpace> vs,
                       const bool throw_on_failure)
{
  Teuchos::RCP<const Thyra_ProductVectorSpace> pvs;
  pvs = Teuchos::rcp_dynamic_cast<const Thyra_ProductVectorSpace>(vs,throw_on_failure);
  return pvs;
}

Teuchos::RCP<Thyra_ProductVector>
getProductVector (const Teuchos::RCP<Thyra_Vector> v,
                  const bool throw_on_failure)
{
  Teuchos::RCP<Thyra_ProductVector> pv;
  pv = Teuchos::rcp_dynamic_cast<Thyra_ProductVector>(v,throw_on_failure);
  return pv;
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
  return pmv;
}

Teuchos::RCP<const Thyra_ProductMultiVector>
getConstProductMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv,
                            const bool throw_on_failure)
{
  Teuchos::RCP<const Thyra_ProductMultiVector> pmv;
  pmv = Teuchos::rcp_dynamic_cast<const Thyra_ProductMultiVector>(mv,throw_on_failure);
  return pmv;
}

} // namespace Albany
