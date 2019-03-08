#include "Albany_ThyraUtils.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_Utils.hpp"

#include "Petra_Converters.hpp"

#include "Thyra_VectorStdOps.hpp"
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DefaultSpmdVector.hpp"

#include "Teuchos_CompilerCodeTweakMacros.hpp"

#if defined(ALBANY_EPETRA)
#include "AztecOO_ConditionNumber.h"
#include "Albany_EpetraThyraUtils.hpp"
#include <type_traits>
#endif

namespace Albany
{

// ========= Vector Spaces utilities ========= //

Teuchos::RCP<const Thyra_VectorSpace>
createLocallyReplicatedVectorSpace(const int size, const Teuchos::RCP<const Teuchos_Comm> comm)
{
  auto comm_thyra = Thyra::convertTpetraToThyraComm(comm);
  return Thyra::locallyReplicatedDefaultSpmdVectorSpace<ST>(comm_thyra,size);
}

GO getGlobalElement (const Teuchos::RCP<const Thyra_VectorSpace>& vs, const LO lid) {
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs,false);
  if (!tmap.is_null()) {
    return tmap->getGlobalElement(lid);
  }

#if defined(ALBANY_EPETRA)
  auto emap = getEpetraBlockMap(vs,false);
  if (!emap.is_null()) {
    return emap->GID(lid);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(-1);
}

LO getLocalElement  (const Teuchos::RCP<const Thyra_VectorSpace>& vs, const GO gid) {
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs,false);
  if (!tmap.is_null()) {
    return tmap->getLocalElement(gid);
  }

#if defined(ALBANY_EPETRA)
  auto emap = getEpetraBlockMap(vs,false);
  if (!emap.is_null()) {
    // Note: simply calling LID(gid) can be ambiguous, if GO!=int and GO!=long long.
    //       Hence, we explicitly cast to whatever has size 64 bits (should *always* be long long, but the if is compile time, so no penalty)
    if (sizeof(GO)==sizeof(int)) {
      return emap->LID(static_cast<int>(gid));
    } else if (sizeof(GO)==sizeof(long long)) {
      return emap->LID(static_cast<long long>(gid));
    } else {
      // We should never reach this point.
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! sizeof(GO) does not match sizeof(int) nor sizeof(long long). "
                                                            "This is an unexpected error. Please, contact developers.\n");
    }
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(-1);
}

Teuchos::RCP<const Thyra_VectorSpace>
removeComponents (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                  const Teuchos::ArrayView<const LO>& local_components)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs,false);
  if (!tmap.is_null()) {
    const LO num_node_lids = tmap->getNodeNumElements();
    const LO num_reduced_node_lids = num_node_lids - local_components.size();
    TEUCHOS_TEST_FOR_EXCEPTION(num_reduced_node_lids<0, std::logic_error, "Error! Cannot remove more components than are actually present.\n");
    Teuchos::Array<Tpetra_GO> reduced_gids(num_reduced_node_lids);
    for (LO lid=0,k=0; lid<num_node_lids; ++lid) {
      if (std::find(local_components.begin(),local_components.end(),lid)==local_components.end()) {
        reduced_gids[k] = tmap->getGlobalElement(lid);
        ++k;
      }
    }

    Tpetra::global_size_t inv_gs = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    Teuchos::RCP<const Tpetra_Map> reduced_map(new Tpetra_Map(inv_gs,reduced_gids().getConst(),tmap->getIndexBase(),tmap->getComm()));

    return createThyraVectorSpace(reduced_map);
  }

#if defined(ALBANY_EPETRA)
  auto emap = getEpetraBlockMap(vs,false);
  if (!emap.is_null()) {
    const LO num_node_lids = emap->NumMyElements();
    const LO num_reduced_node_lids = num_node_lids - local_components.size();
    TEUCHOS_TEST_FOR_EXCEPTION(num_reduced_node_lids<0, std::logic_error, "Error! Cannot remove more components than are actually present.\n");
    Teuchos::Array<Epetra_GO> reduced_gids(num_reduced_node_lids);
    for (LO lid=0,k=0; lid<num_node_lids; ++lid) {
      if (std::find(local_components.begin(),local_components.end(),lid)==local_components.end()) {
        reduced_gids[k] = emap->GID(lid);
        ++k;
      }
    }

    Teuchos::RCP<const Epetra_BlockMap> reduced_map(new Epetra_BlockMap(-1,reduced_gids().size(),reduced_gids.getRawPtr(),1,emap->IndexBase(),emap->Comm()));
    return createThyraVectorSpace(reduced_map);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
}

Teuchos::RCP<const Thyra_VectorSpace>
createSubspace (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                const Teuchos::ArrayView<const LO>& subspace_components)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs,false);
  if (!tmap.is_null()) {
    Teuchos::Array<Tpetra_GO> subspace_gids(subspace_components.size());
    int k = 0;
    for (auto lid : subspace_components) {
      subspace_gids[k] = tmap->getGlobalElement(lid);
      ++k;
    }

    Tpetra::global_size_t inv_gs = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    Teuchos::RCP<const Tpetra_Map> reduced_map(new Tpetra_Map(inv_gs,subspace_gids().getConst(),tmap->getIndexBase(),tmap->getComm()));

    return createThyraVectorSpace(reduced_map);
  }

#if defined(ALBANY_EPETRA)
  auto emap = getEpetraBlockMap(vs,false);
  if (!emap.is_null()) {
    Teuchos::Array<Epetra_GO> subspace_gids(subspace_components.size());
    int k = 0;
    for (auto lid : subspace_components) {
      subspace_gids[k] = emap->GID(lid);
      ++k;
    }

    Teuchos::RCP<const Epetra_BlockMap> reduced_map(new Epetra_BlockMap(-1,subspace_gids().size(),subspace_gids.getRawPtr(),1,emap->IndexBase(),emap->Comm()));
    return createThyraVectorSpace(reduced_map);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
}

// ========= Thyra_LinearOp utilities ========= //

bool isFillActive (const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    return tmat->isFillActive();
  }

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    return !emat->Filled();
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    // Nothing to do in Epetra. As long as you only need to change the values (not the graph),
    // Epetra already let's you do it on a filled matrix
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    emat->FillComplete();
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    emat->PutScalar(value);
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    emat->ExtractDiagonalCopy(*Albany::getEpetraVector(diag,true));
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    auto numEntries = emat->NumMyEntries(lrow);
    indices.resize(numEntries);
    values.resize(numEntries);
    emat->ExtractMyRowCopy(lrow,numEntries,numEntries,values.getRawPtr(),indices.getRawPtr());
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    emat->SumIntoMyValues(lrow,indices.size(),values.getRawPtr(),indices.getRawPtr());
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    emat->ReplaceMyValues(lrow,indices.size(),values.getRawPtr(),indices.getRawPtr());
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

double computeConditionNumber (const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  double condest = std::numeric_limits<double>::quiet_NaN();

#ifdef ALBANY_EPETRA
  // Allow failure, since we don't know what the underlying linear algebra is
  Teuchos::RCP<const Epetra_CrsMatrix> emat;
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    Petra::Converter converter(tmat->getComm());

    emat = Petra::TpetraCrsMatrix_To_EpetraCrsMatrix(tmat,converter.commE_);
  }

  // Try epetra if tpetra didn't work
  if (emat.is_null()) {
    emat = getConstEpetraMatrix(lop,false);
  }

  if (emat.is_null()) {
    // If all the tries above are unsuccessful, throw an error.
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
  } else {
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
    condest = conditionEstimator.getConditionNumber();
    return condest;
  }
#else
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Condition number estimation requires ALBANY_EPETRA.\n");
  // Suppress compiler warning for unused argument
  (void) lop;
#endif

  // Dummy return value to silence compiler warning
  return condest;
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

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION ((!std::is_same<PHX::Device::memory_space,Kokkos::HostSpace>::value),
                                std::logic_error,
                                "Error! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");

    Teuchos::Array<LO> row_idx(emat->NumMyNonzeros()), col_idx(emat->NumMyNonzeros());
    Teuchos::Array<ST> vals(emat->NumMyNonzeros());
    int numEntries;
    int* idx_tmp;
    ST* vals_tmp;
    for (int irow=0, k=0; irow<emat->NumMyRows(); ++irow) {
      emat->ExtractMyRowView(irow,numEntries,vals_tmp,idx_tmp);
      for (int icol=0; icol<numEntries; ++icol, ++k) {
        row_idx[k] = irow;
        col_idx[k] = idx_tmp[icol];
        vals[k] = vals_tmp[icol];
      }
    }
    DeviceLocalMatrix<ST> data("Epetra device data",
                               emat->NumMyRows(), emat->NumMyCols(), emat->NumMyNonzeros(),
                               vals.getRawPtr(), row_idx.getRawPtr(), col_idx.getRawPtr());
    return data;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION ((!std::is_same<PHX::Device::memory_space,Kokkos::HostSpace>::value),
                                std::logic_error,
                                "Error! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");

    Teuchos::Array<LO> row_idx(emat->NumMyNonzeros()), col_idx(emat->NumMyNonzeros());
    Teuchos::Array<ST> vals(emat->NumMyNonzeros());
    int numEntries;
    int* idx_tmp;
    ST* vals_tmp;
    for (int irow=0, k=0; irow<emat->NumMyRows(); ++irow) {
      emat->ExtractMyRowView(irow,numEntries,vals_tmp,idx_tmp);
      for (int icol=0; icol<numEntries; ++icol, ++k) {
        row_idx[k] = irow;
        col_idx[k] = idx_tmp[icol];
        vals[k] = vals_tmp[icol];
      }
    }
    DeviceLocalMatrix<ST> data("Epetra device data",
                               emat->NumMyRows(), emat->NumMyCols(), emat->NumMyNonzeros(),
                               vals.getRawPtr(), row_idx.getRawPtr(), col_idx.getRawPtr());
    return data;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceLocalMatrix<ST> dummy;
  return dummy;
}

// ========= Thyra_Vector utilities ========== //

Teuchos::ArrayRCP<ST> getNonconstLocalData (const Teuchos::RCP<Thyra_Vector>& v)
{
  Teuchos::ArrayRCP<ST> vals;

  // Allow failure, since we don't know what the underlying linear algebra is
  // Note: we do tpetra separately since it need to handle device/copy sync.
  //       everything else, we assume it inherits from SpmdVectorBase.
  auto tv = getTpetraVector(v,false);
  if (!tv.is_null()) {
    // Tpetra
    vals = tv->get1dViewNonConst();
  } else {
    // Thyra::SpmdVectorBase
    auto spmd_v = Teuchos::rcp_dynamic_cast<Thyra::SpmdVectorBase<ST>>(v);
    if (!spmd_v.is_null()) {
      spmd_v->getNonconstLocalData(Teuchos::outArg(vals));
    } else {
      // If all the tries above are unsuccessful, throw an error.
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
    }
  }

  return vals;
}

Teuchos::ArrayRCP<const ST> getLocalData (const Teuchos::RCP<const Thyra_Vector>& v)
{
  Teuchos::ArrayRCP<const ST> vals;

  // Allow failure, since we don't know what the underlying linear algebra is
  // Note: we do tpetra separately since it need to handle device/copy sync.
  //       everything else, we assume it inherits from SpmdVectorBase.
  auto tv = getConstTpetraVector(v,false);
  if (!tv.is_null()) {
    // Tpetra
    vals = tv->get1dView();
  } else {
    // Thyra::SpmdVectorBase
    auto spmd_v = Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorBase<ST>>(v);
    if (!spmd_v.is_null()) {
      spmd_v->getLocalData(Teuchos::outArg(vals));
    } else {
      // If all the tries above are unsuccessful, throw an error.
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
    }
  }

  return vals;
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

#if defined(ALBANY_EPETRA)
  auto evec = getConstEpetraVector(v,false);
  if (!evec.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION ((!std::is_same<PHX::Device::memory_space,Kokkos::HostSpace>::value),
                                std::logic_error,
                                "Error! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");
    DeviceView1d<const ST> data( evec->Values(), evec->MyLength() );
    return data;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto evec = getEpetraVector(v,false);
  if (!evec.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION ((!std::is_same<PHX::Device::memory_space,Kokkos::HostSpace>::value),
                                std::logic_error,
                                "Error! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");
    DeviceView1d<ST> data( evec->Values(), evec->MyLength() );
    return data;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

ST mean (const Teuchos::RCP<const Thyra_Vector>& v) {
  return Thyra::sum(*v)/v->space()->dim();
}

Teuchos::Array<ST> means (const Teuchos::RCP<const Thyra_MultiVector>& mv) {
  const int numVecs = mv->domain()->dim();
  Teuchos::Array<ST> vals(numVecs);
  for (int i=0; i<numVecs; ++i) {
    vals[i] = mean(mv->col(i));
  }

  return vals;
}

// ======== I/O utilities ========= //

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

#if defined(ALBANY_EPETRA)
  auto ev = getConstEpetraVector(v,false);
  if (!ev.is_null()) {
    ev->Print(*out.getOStream());
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
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

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(op,false);
  if (!emat.is_null()) {
    emat->Print(*out.getOStream());
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
}

// ========= Matrix Market utilities ========== //

// These routines implement a specialization of the template functions declared in Albany_Utils.hpp

template<>
void
writeMatrixMarket<const Thyra_Vector>(
    const Teuchos::RCP<const Thyra_Vector>& v,
    const std::string& prefix,
    const int counter)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tv = getConstTpetraVector(v,false);
  if (!tv.is_null()) {
    writeMatrixMarket(tv,prefix,counter);
    return;
  }

#if defined(ALBANY_EPETRA)
  auto ev = getConstEpetraVector(v,false);
  if (!ev.is_null()) {
    // TODO: avoid petra conversion, and call EpetraExt I/O directly
    tv = Petra::EpetraVector_To_TpetraVectorConst(*ev,createTeuchosCommFromEpetraComm(ev->Comm()));
    writeMatrixMarket(tv,prefix,counter);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
}

template<>
void
writeMatrixMarket<Thyra_Vector>(
    const Teuchos::RCP<Thyra_Vector>& v,
    const std::string& prefix,
    const int counter)
{
  writeMatrixMarket(v.getConst(),prefix,counter);
}

template<>
void
writeMatrixMarket<const Thyra_MultiVector>(
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

#if defined(ALBANY_EPETRA)
  auto emv = getConstEpetraMultiVector(mv,false);
  if (!emv.is_null()) {
    // TODO: avoid petra conversion, and call EpetraExt I/O directly
    tmv = Petra::EpetraMultiVector_To_TpetraMultiVector(*emv,createTeuchosCommFromEpetraComm(emv->Comm()));
    writeMatrixMarket(tmv,prefix,counter);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
}

template<>
void
writeMatrixMarket<Thyra_MultiVector>(
    const Teuchos::RCP<Thyra_MultiVector>& mv,
    const std::string& prefix,
    const int counter)
{
  writeMatrixMarket(mv.getConst(),prefix,counter);
}

template<>
void
writeMatrixMarket<const Thyra_LinearOp>(
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

#if defined(ALBANY_EPETRA)
  auto eA = getConstEpetraMatrix(A,false);
  if (!eA.is_null()) {
    // TODO: avoid petra conversion, and call EpetraExt I/O directly
    tA = Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(*eA,createTeuchosCommFromEpetraComm(eA->Comm()));
    writeMatrixMarket(tA,prefix,counter);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

template<>
void
writeMatrixMarket<Thyra_LinearOp>(
    const Teuchos::RCP<Thyra_LinearOp>& A,
    const std::string& prefix,
    const int counter)
{
  writeMatrixMarket(A.getConst(),prefix,counter);
}

// These routines implement a specialization of the template functions declared in Albany_Utils.hpp
template<>
void
writeMatrixMarket<const Thyra_VectorSpace>(
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

#if defined(ALBANY_EPETRA)
  auto em = getEpetraBlockMap(vs,false);
  if (!em.is_null()) {
    // TODO: avoid petra conversion, and call EpetraExt I/O directly
    tm = Petra::EpetraMap_To_TpetraMap(*em,createTeuchosCommFromEpetraComm(em->Comm()));
    writeMatrixMarket(tm,prefix,counter);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");
}

template<>
void
writeMatrixMarket<Thyra_VectorSpace>(
    const Teuchos::RCP<Thyra_VectorSpace>& vs,
    const std::string& prefix,
    const int counter)
{
  writeMatrixMarket(vs.getConst(),prefix,counter);
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
