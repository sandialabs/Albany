#include "Albany_ThyraUtils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraCrsMatrixFactory.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_Utils.hpp"
#include "Albany_Macros.hpp"

#include "Petra_Converters.hpp"

#include "Thyra_VectorStdOps.hpp"
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DefaultSpmdVector.hpp"

#include "Teuchos_CompilerCodeTweakMacros.hpp"
#include "Teuchos_RCP.hpp"

#if defined(ALBANY_EPETRA)
#include "Thyra_EpetraLinearOp.hpp"
#include "AztecOO_ConditionNumber.h"
#include "Albany_EpetraThyraUtils.hpp"
#include "Epetra_LocalMap.h"
#include <type_traits>
#endif

namespace Albany
{

// ========= Vector Spaces utilities ========= //

Teuchos::RCP<const Thyra_VectorSpace>
createLocallyReplicatedVectorSpace(const int size, const Teuchos::RCP<const Teuchos_Comm> comm)
{
  auto bt = build_type();
  switch (bt) {
#ifdef ALBANY_EPETRA
    case BuildType::Epetra:
    {
      Teuchos::RCP<const Epetra_BlockMap> emap( new Epetra_LocalMap(size,0,*createEpetraCommFromTeuchosComm(comm)) );
      return createThyraVectorSpace(emap);
      break;
    }
#endif
    case BuildType::Tpetra:
    {
      Teuchos::RCP<const Tpetra_Map> tmap( new Tpetra_Map(size,0,comm,Tpetra::LocalGlobal::LocallyReplicated) );
      return createThyraVectorSpace(tmap);
      break;
    }
    default:
    {
      auto comm_thyra = createThyraCommFromTeuchosComm(comm);
      return Thyra::locallyReplicatedDefaultSpmdVectorSpace<ST>(comm_thyra,size);
    }
  }

  TEUCHOS_UNREACHABLE_RETURN (Teuchos::null);
}

Teuchos::RCP<const Teuchos_Comm> getComm (const Teuchos::RCP<const Thyra_VectorSpace>& vs)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs,false);
  if (!tmap.is_null()) {
    return tmap->getComm(); 
  }
#if defined(ALBANY_EPETRA)
  auto emap = getEpetraBlockMap(vs,false);
  if (!emap.is_null()) {
    return createTeuchosCommFromEpetraComm(emap->Comm()); 
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getComm! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

}

GO getMaxAllGlobalIndex(const Teuchos::RCP<const Thyra_VectorSpace>& vs) {
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs,false);
  if (!tmap.is_null()) {
    return tmap->getMaxAllGlobalIndex();
  }
#if defined(ALBANY_EPETRA)
  auto emap = getEpetraBlockMap(vs,false);
  if (!emap.is_null()) {
    return static_cast<GO>(emap->MaxElementSize()); 
  }
#endif
  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getMaxAllGlobalIndex! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");
}

Teuchos::Array<GO> getGlobalElements  (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                                       const Teuchos::ArrayView<const LO>& lids)
{
  auto indexer = createGlobalLocalIndexer(vs);
  Teuchos::Array<GO> gids(lids.size());
  for (LO i=0; i<lids.size(); ++i) {
    gids[i] = indexer->getGlobalElement(lids[i]);
  }
  return gids;
}

Teuchos::Array<LO> getLocalElements  (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                                      const Teuchos::ArrayView<const GO>& gids)
{
  auto indexer = createGlobalLocalIndexer(vs);
  Teuchos::Array<LO> lids(gids.size());
  for (LO i=0; i<gids.size(); ++i) {
    lids[i] = indexer->getLocalElement(gids[i]);
  }
  return lids;
}

void getGlobalElements (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                        const Teuchos::ArrayView<GO>& gids)
{
  auto indexer = createGlobalLocalIndexer(vs);
  const LO localDim = indexer->getNumLocalElements();
  TEUCHOS_TEST_FOR_EXCEPTION(gids.size()!=localDim, std::runtime_error, "Error! ArrayView for gids not properly dimensioned.\n");

  for (LO i=0; i<localDim; ++i) {
    gids[i] = indexer->getGlobalElement(i);
  }
}

Teuchos::Array<GO> getGlobalElements (const Teuchos::RCP<const Thyra_VectorSpace>& vs)
{
  Teuchos::Array<GO> gids(getLocalSubdim(vs));
  getGlobalElements(vs,gids());
  return gids;
}

LO getLocalSubdim( const Teuchos::RCP<const Thyra_VectorSpace>& vs)
{
  auto spmd_vs = getSpmdVectorSpace(vs);
  return spmd_vs->localSubDim();
} 

bool sameAs (const Teuchos::RCP<const Thyra_VectorSpace>& vs1,
             const Teuchos::RCP<const Thyra_VectorSpace>& vs2)
{
  auto tmap1 = getTpetraMap(vs1,false);
  if (!tmap1.is_null()) {
    // We don't allow two vs with different linear algebra back ends
    auto tmap2 = getTpetraMap(vs2,true);
    return tmap1->isSameAs(*tmap2);
  }
#if defined(ALBANY_EPETRA)
  auto emap1 = getEpetraBlockMap(vs1,false);
  if (!emap1.is_null()) {
    // We don't allow two vs with different linear algebra back ends
    auto emap2 = getEpetraBlockMap(vs2,true);
    return emap2->SameAs(*emap1);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in sameAs! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(false);
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
    TEUCHOS_TEST_FOR_EXCEPTION(num_reduced_node_lids<0, std::logic_error, "Error in removeComponents! Cannot remove more components than are actually present.\n");
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
    TEUCHOS_TEST_FOR_EXCEPTION(num_reduced_node_lids<0, std::logic_error, "Error in removeComponents! Cannot remove more components than are actually present.\n");
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in removeComponents! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in createSubspace! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(Teuchos::null);
}

// Create a vector space, given the ids of the space components
Teuchos::RCP<const Thyra_SpmdVectorSpace>
createVectorSpace (const Teuchos::RCP<const Teuchos_Comm>& comm,
                   const Teuchos::ArrayView<const GO>& gids,
                   const GO globalDim)
{
  auto bt = build_type();
  const GO invalid = Teuchos::OrdinalTraits<GO>::invalid();
  if (bt == BuildType::Epetra) {
#ifdef ALBANY_EPETRA
    auto ecomm = createEpetraCommFromTeuchosComm(comm);
    Teuchos::RCP<const Epetra_BlockMap> emap;
    const Epetra_GO numGlobalElements = (globalDim==invalid) ? -1 : static_cast<Epetra_GO>(globalDim);
    if (sizeof(GO)==sizeof(Epetra_GO)) {
      // Same size, different type names. A reinterpret_cast is safe
      const Epetra_GO* egids = reinterpret_cast<const Epetra_GO*>(gids.getRawPtr());
      emap = Teuchos::rcp( new Epetra_BlockMap(numGlobalElements,gids.size(),egids,1,0,*ecomm) );
    } else {
      // The types have a different size. Need to copy GO's into Epetra_GO's
      Teuchos::Array<Epetra_GO> egids(gids.size());
      const GO max_safe_gid = static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max());
      for (int i=0; i<gids.size(); ++i) {
        ALBANY_EXPECT(gids[i]<=max_safe_gid, "Error in createVectorSpace! Input gids exceed Epetra_GO ranges.\n");
        egids[i] = static_cast<Epetra_GO>(gids[i]);
      }
      (void) max_safe_gid;
      emap = Teuchos::rcp( new Epetra_BlockMap(numGlobalElements,gids.size(),egids.getRawPtr(),1,0,*ecomm) );
    }
    return createThyraVectorSpace(emap);
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in createVectorSpace! Epetra build not supported.\n");
#endif
  } else if (bt == BuildType::Tpetra) {
    auto gsi = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const decltype(gsi) numGlobalElements = (globalDim==invalid) ? gsi : static_cast<Tpetra_GO>(globalDim);
    Teuchos::ArrayView<const Tpetra_GO> tgids(reinterpret_cast<const Tpetra_GO*>(gids.getRawPtr()),gids.size());
    Teuchos::RCP<const Tpetra_Map> tmap = Teuchos::rcp( new Tpetra_Map(numGlobalElements,tgids,0,comm) );
    return createThyraVectorSpace(tmap);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in createVectorSpace! Invalid or unsupported build type.\n");
  }
}

Teuchos::RCP<const Thyra_VectorSpace>
createVectorSpacesIntersection(const Teuchos::RCP<const Thyra_VectorSpace>& vs1,
                               const Teuchos::RCP<const Thyra_VectorSpace>& vs2,
                               const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  auto gids1 = getGlobalElements(vs1);
  auto gids2 = getGlobalElements(vs2);
  std::sort(gids1.begin(),gids1.end());
  std::sort(gids2.begin(),gids2.end());

  const auto min_size = std::min(gids1.size(),gids2.size());

  Teuchos::Array<GO> gids(min_size);
  const auto it = std::set_intersection(gids1.begin(),gids1.end(),gids2.begin(),gids2.end(),gids.begin());
  gids.resize(std::distance(gids.begin(),it));

  return createVectorSpace(comm,gids);
}

Teuchos::RCP<const Thyra_VectorSpace>
createVectorSpacesDifference (const Teuchos::RCP<const Thyra_VectorSpace>& vs1,
                              const Teuchos::RCP<const Thyra_VectorSpace>& vs2,
                              const Teuchos::RCP<const Teuchos_Comm>& comm)
{
  auto gids1 = getGlobalElements(vs1);
  auto gids2 = getGlobalElements(vs2);
  std::sort(gids1.begin(),gids1.end());
  std::sort(gids2.begin(),gids2.end());

  Teuchos::Array<GO> gids;
  std::set_difference(gids1.begin(),gids1.end(),
                      gids2.begin(),gids2.end(),
                      std::back_inserter(gids));

  return createVectorSpace(comm,gids);
}

// ========= Thyra_LinearOp utilities ========= //

Teuchos::RCP<const Thyra_VectorSpace>
getColumnSpace (const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    return createThyraVectorSpace(tmat->getColMap());
  }

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    Teuchos::RCP<const Epetra_BlockMap> col_map = Teuchos::rcpFromRef(emat->ColMap());
    return createThyraVectorSpace(col_map);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getColumnSpace! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return Teuchos::null;
}

Teuchos::RCP<const Thyra_VectorSpace>
getRowSpace (const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    return createThyraVectorSpace(tmat->getRowMap());
  }

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    Teuchos::RCP<const Epetra_BlockMap> row_map = Teuchos::rcpFromRef(emat->RowMap());
    return createThyraVectorSpace(row_map);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getRowSpace! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return Teuchos::null;
}

std::size_t
getNumEntriesInLocalRow (const Teuchos::RCP<const Thyra_LinearOp>& lop, const LO lrow)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    return tmat->getNumEntriesInLocalRow(lrow);
  }

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    return emat->NumMyEntries(lrow);
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getNumEntriesInLocalRow! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return Teuchos::null;

}


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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in isFillActive! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  return false;
}

bool isFillComplete (const Teuchos::RCP<const Thyra_LinearOp>& lop)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    return tmat->isFillComplete();
  }

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    return emat->Filled();
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in isFillComplete! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in resumeFill! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
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
    emat->OptimizeStorage();  // This allows to extract data with 'ExtractCrsDataPointers
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in fillComplete! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in assign! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void getDiagonalCopy (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                      Teuchos::RCP<Thyra_Vector>& diag)
{
  // Diagonal makes sense only for (globally) square operators.
  // From Thyra, we can't check the global ids of the range/domain vector spaces,
  // but at least we can check that they have the same (global) dimension.
  TEUCHOS_TEST_FOR_EXCEPTION(lop->range()->dim()!=lop->domain()->dim(), std::logic_error,
                              "Error in getDiagonalCopy! Attempt to take the diagonal of a non-square operator.\n");

  // If diag is not created, do it.
  if (diag.is_null()) {
    diag = Thyra::createMember(lop->range());
  }

  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    auto tvec = *Albany::getTpetraVector(diag,true);
    tvec.clear_sync_state();
    tmat->getLocalDiagCopy(tvec);
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getDiagonalCopy! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void scale (const Teuchos::RCP<Thyra_LinearOp>& lop, const ST val) 
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    tmat->scale(val); 
    return; 
  }
#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    emat->Scale(val); 
    return; 
  }
#endif
  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in scale! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getLocalRowValues! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

int addToLocalRowValues (const Teuchos::RCP<Thyra_LinearOp>& lop,
                          const LO lrow,
                          const Teuchos::ArrayView<const LO> indices,
                          const Teuchos::ArrayView<const ST> values)
{
  //The following is an integer error code, to be returned by this 
  //routine if something doesn't go right.  0 means success, 1 means failure 
  int integer_error_code = 0; 
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    auto returned_val = tmat->sumIntoLocalValues(lrow,indices,values);
    //std::cout << "IKT returned_val, indices size = " << returned_val << ", " << indices.size() << std::endl; 
    ALBANY_ASSERT(returned_val != -1 , "Error: addToLocalRowValues returned -1, meaning linear op is not fillActive \n" 
                       << "or does not have an underlying non-null static graph!\n"); 
    //Tpetra's replaceLocalValues routine returns the number of indices for which values were actually replaced; the number of "correct" indices.
    //This should be size of indices array.  Therefore if returned_val != indices.size() something went wrong 
    if (returned_val != indices.size()) integer_error_code = 1; 
    return integer_error_code; 
  }

#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    //Epetra's ReplaceMyValues routine returns integer error code, set to 0 if successful, set to 1 if one or more indices are not 
    //associated with the calling processor.  We can just return that value for the Epetra case. 
    integer_error_code = emat->SumIntoMyValues(lrow,indices.size(),values.getRawPtr(),indices.getRawPtr());
    return integer_error_code;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in addToLocalRowValues! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void insertGlobalValues (const Teuchos::RCP<Thyra_LinearOp>& lop,
                         const GO grow,
                         const Teuchos::ArrayView<const GO> cols,
                         const Teuchos::ArrayView<const ST> values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    const Tpetra_GO tgrow = grow;
    Teuchos::ArrayView<const Tpetra_GO> tcols(reinterpret_cast<const Tpetra_GO*>(cols.getRawPtr()),cols.size());
    tmat->insertGlobalValues(tgrow, tcols, values); 
    return; 
  }
#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    const Epetra_GO egrow = grow;
    if (sizeof(GO)==sizeof(Epetra_GO)) {
      Teuchos::ArrayView<const Epetra_GO> ecols(reinterpret_cast<const Epetra_GO*>(ecols.getRawPtr()),ecols.size());
      emat->InsertGlobalValues(egrow, ecols.size(), values.getRawPtr(), ecols.getRawPtr());
    }
    else {
      // Cannot reinterpret cast. Need to copy gids into Epetra_GO array
      Teuchos::Array<Epetra_GO> ecols(cols.size());
      const GO max_safe_col = static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max());
      for (int i=0; i<cols.size(); ++i) {
        ALBANY_EXPECT(cols[i]<=max_safe_col, "Error in insertGlobalValues! Input cols exceed Epetra_GO ranges.\n");
        ecols[i] = static_cast<Epetra_GO>(cols[i]);
      }
      ALBANY_EXPECT(grow<=max_safe_col, "Error in insertGlobalValues! Input grow exceeds Epetra_GO ranges.\n");
      (void) max_safe_col;
      emat->InsertGlobalValues(egrow, ecols.size(), values.getRawPtr(), ecols.getRawPtr());
    }
    return; 
  }
#endif
}

void replaceGlobalValues (const Teuchos::RCP<Thyra_LinearOp>& lop,
                             const GO gid,
                             const Teuchos::ArrayView<const GO> indices,
                             const Teuchos::ArrayView<const ST> values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    const Tpetra_GO tgid = gid;
    Teuchos::ArrayView<const Tpetra_GO> tindices(reinterpret_cast<const Tpetra_GO*>(indices.getRawPtr()),indices.size());
    tmat->replaceGlobalValues(tgid,tindices,values);
    return;
  }
#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    const Epetra_GO egid = gid;
    if (sizeof(GO)==sizeof(Epetra_GO)) {
      Teuchos::ArrayView<const Epetra_GO> eindices(reinterpret_cast<const Epetra_GO*>(indices.getRawPtr()),indices.size());
      emat->ReplaceGlobalValues(egid, eindices.size(), values.getRawPtr(), eindices.getRawPtr());
    }
    else {
      // Cannot reinterpret cast. Need to copy gids into Epetra_GO array
      Teuchos::Array<Epetra_GO> eindices(indices.size());
      const GO max_safe_index = static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max());
      for (int i=0; i<indices.size(); ++i) {
        ALBANY_EXPECT(indices[i]<=max_safe_index, "Error in replaceGlobalValues! Input indices exceed Epetra_GO ranges.\n");
        eindices[i] = static_cast<Epetra_GO>(indices[i]);
      }
      ALBANY_EXPECT(gid<=max_safe_index, "Error in replaceGlobalValues! Input grow exceeds Epetra_GO ranges.\n");
      (void) max_safe_index;
      emat->ReplaceGlobalValues(egid, eindices.size(), values.getRawPtr(), eindices.getRawPtr());
    }
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in replaceGlobalValues! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

int addToGlobalRowValues (const Teuchos::RCP<Thyra_LinearOp>& lop,
                          const GO grow,
                          const Teuchos::ArrayView<const GO> indices,
                          const Teuchos::ArrayView<const ST> values)
{
  //The following is an integer error code, to be returned by this 
  //routine if something doesn't go right.  0 means success, 1 means failure 
  int integer_error_code = 0; 
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    const Tpetra_GO tgrow = grow;
    Teuchos::ArrayView<const Tpetra_GO> tindices(reinterpret_cast<const Tpetra_GO*>(indices.getRawPtr()),indices.size());
    auto returned_val = tmat->sumIntoGlobalValues(tgrow,tindices,values);
    //std::cout << "IKT returned_val, indices size = " << returned_val << ", " << indices.size() << std::endl; 
    ALBANY_ASSERT(returned_val != -1, "Error: addToGlobalRowValues returned -1, meaning linear op is not fillActive \n" 
                       << "or does not have an underlying non-null static graph!\n"); 
    //Tpetra's replaceGlobalValues routine returns the number of indices for which values were actually replaced; the number of "correct" indices.
    //This should be size of indices array.  Therefore if returned_val != indices.size() something went wrong 
    if (returned_val != indices.size()) integer_error_code = 1; 
    return integer_error_code; 
  }

#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    //Epetra's ReplaceGlobalValues routine returns integer error code, set to 0 if successful, set to 1 if one or more indices are not 
    //associated with the calling processor.  We can just return that value for the Epetra case. 
    const Epetra_GO egrow = grow;
    if (sizeof(GO)==sizeof(Epetra_GO)) {
      Teuchos::ArrayView<const Epetra_GO> eindices(reinterpret_cast<const Epetra_GO*>(indices.getRawPtr()),indices.size());
      integer_error_code = emat->SumIntoGlobalValues(egrow, eindices.size(), values.getRawPtr(), eindices.getRawPtr());
    }
    else {
      // Cannot reinterpret cast. Need to copy gids into Epetra_GO array
      Teuchos::Array<Epetra_GO> eindices(indices.size());
      const GO max_safe_index = static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max());
      for (int i=0; i<indices.size(); ++i) {
        ALBANY_EXPECT(indices[i]<=max_safe_index, "Error in addToGlobalRowValues! Input indices exceed Epetra_GO ranges.\n");
        eindices[i] = static_cast<Epetra_GO>(indices[i]);
      }
      ALBANY_EXPECT(grow<=max_safe_index, "Error in addToGlobalRowValues! Input grow exceeds Epetra_GO ranges.\n");
      (void) max_safe_index;
      integer_error_code = emat->SumIntoGlobalValues(egrow, eindices.size(), values.getRawPtr(), eindices.getRawPtr());
    }
    return integer_error_code;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in addToGlobalRowValues! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in setLocalRowValues! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
}

void setLocalRowValues (const Teuchos::RCP<Thyra_LinearOp>& lop,
                        const LO lrow,
                        const Teuchos::ArrayView<const ST> values)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    Teuchos::ArrayView<const LO> indices;
    tmat->getGraph()->getLocalRowView(lrow,indices);
    TEUCHOS_TEST_FOR_EXCEPTION(indices.size()!=values.size(), std::logic_error,
                               "Error! This routine is meant for setting *all* values in a row, "
                               "but the length of the input values array does not match the number of indices in the local row.\n");
    tmat->replaceLocalValues(lrow,indices,values);
    return;
  }

#if defined(ALBANY_EPETRA)
  auto emat = getEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    int numIndices;
    int* indices;
    emat->Graph().ExtractMyRowView (lrow, numIndices, indices);
    TEUCHOS_TEST_FOR_EXCEPTION(numIndices!=values.size(), std::logic_error,
                               "Error! This routine is meant for setting *all* values in a row, "
                               "but the length of the input values array does not match the number of indices in the local row.\n");
    emat->ReplaceMyValues(lrow,numIndices,values.getRawPtr(),indices);
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in setLocalRowValues! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

}

int getGlobalMaxNumRowEntries (const Teuchos::RCP<const Thyra_LinearOp>& lop) 
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmat = getConstTpetraMatrix(lop,false);
  if (!tmat.is_null()) {
    auto return_value = tmat->getGlobalMaxNumRowEntries();
    return return_value; 
  }

#if defined(ALBANY_EPETRA)
  auto emat = getConstEpetraMatrix(lop,false);
  if (!emat.is_null()) {
    auto return_value = emat->GlobalMaxNumEntries(); 
    return return_value;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getGlobalMaxNumRowEntries! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");

}

//The following routine creates a one-to-one version of the given Map where each GID lives on only one process. 
//Therefore it is an owned (unique) map.
Teuchos::RCP<const Thyra_VectorSpace>
createOneToOneVectorSpace (const Teuchos::RCP<const Thyra_VectorSpace> vs)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs,false);
  if (!tmap.is_null()) {
    const Teuchos::RCP<const Tpetra_Map> map = Tpetra::createOneToOne(tmap);
    return createThyraVectorSpace(map);
  }
#if defined(ALBANY_EPETRA)
  auto emap = getEpetraMap(vs,false);
  if (!emap.is_null()) {
    const auto map = Epetra_Util::Create_OneToOne_Map(*emap);
    const Teuchos::RCP<const Epetra_BlockMap> map_rcp = Teuchos::rcpFromRef(map); 
    return createThyraVectorSpace(map_rcp);
  }
#endif
  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in createOneToOneVectorSpace! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");
}

Teuchos::RCP<const Thyra_LinearOp>
buildRestrictionOperator (const Teuchos::RCP<const Thyra_VectorSpace>& space,
                          const Teuchos::RCP<const Thyra_VectorSpace>& subspace)
{
  // In the process, verify the that subspace is a subspace of space
  auto space_indexer    = createGlobalLocalIndexer(space);
  auto subspace_indexer = createGlobalLocalIndexer(subspace);

  ThyraCrsMatrixFactory factory(space,subspace);

  const int localSubDim = subspace_indexer->getNumLocalElements();
  for (LO lid=0; lid<localSubDim; ++lid) {
    const GO gid = subspace_indexer->getGlobalElement(lid);
    TEUCHOS_TEST_FOR_EXCEPTION (space_indexer->isLocallyOwnedElement(gid), std::logic_error,
                                "Error in buildRestrictionOperator! The input 'subspace' is not a subspace of the input 'space'.\n");
    factory.insertGlobalIndices(gid,Teuchos::arrayView(&gid,1));
  }

  factory.fillComplete();
  Teuchos::RCP<Thyra_LinearOp> P = factory.createOp();
  assign(P,1.0);

  return P;
}

Teuchos::RCP<const Thyra_LinearOp>
buildProlongationOperator (const Teuchos::RCP<const Thyra_VectorSpace>& space,
                           const Teuchos::RCP<const Thyra_VectorSpace>& subspace)
{
  // In the process, verify the that subspace is a subspace of space
  auto space_indexer    = createGlobalLocalIndexer(space);
  auto subspace_indexer = createGlobalLocalIndexer(subspace);

  ThyraCrsMatrixFactory factory(subspace,space);

  const int localSubDim = subspace_indexer->getNumLocalElements();
  for (LO lid=0; lid<localSubDim; ++lid) {
    const GO gid = subspace_indexer->getGlobalElement(lid);
    TEUCHOS_TEST_FOR_EXCEPTION (space_indexer->isLocallyOwnedElement(gid), std::logic_error,
                                "Error in buildProlongationOperator! The input 'subspace' is not a subspace of the input 'space'.\n");
    factory.insertGlobalIndices(gid,Teuchos::arrayView(&gid,1));
  }

  factory.fillComplete();
  Teuchos::RCP<Thyra_LinearOp> P = factory.createOp();
  assign(P,1.0);

  return P;
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
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in computeConditionNumber! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
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

DeviceLocalMatrix<const ST> getDeviceData (Teuchos::RCP<const Thyra_LinearOp>& lop)
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
                                "Error in getDeviceData! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");

    // If you want the output DeviceLocalMatrix to have view semantic on the matrix values,
    // you need to use the constructor that 'views' the input arrays.
    // So we need to create views unmanaged, which need to view the matrix data.
    // WARNING: This is *highly* relying on Epetra_CrsMatrix internal storage.
    //          More precisely, I'm not even sure this routine could be fixed
    //          if Epetra_CrsMatrix changes the internal storage scheme.

    using StaticGraphType = DeviceLocalMatrix<ST>::staticcrsgraph_type;
    using size_type = StaticGraphType::size_type;
    TEUCHOS_TEST_FOR_EXCEPTION (sizeof(size_type)!=sizeof(LO), std::runtime_error,
                                "Error in getDeviceData! Extracting local data from an Epetra_CrsMatrix is safe only as long as "
                                "the size of Kokkos::HostSpace::size_type equals sizeof(LO).\n");

    // Some data from the matrix
    const int numMyRows = emat->NumMyRows();
    const int numMyCols = emat->NumMyCols();
    const int numMyNonzeros = emat->NumMyNonzeros();

    // Grab the data
    LO* row_map;
    LO* indices;
    ST* values;
    int err_code = emat->ExtractCrsDataPointers(row_map,indices,values);
    ALBANY_EXPECT(err_code==0, "Error in getDeviceData! Something went wrong while extracting Epetra_CrsMatrix local data pointers.\n");
    (void) err_code;
    Teuchos::ArrayRCP<size_type> row_map_size_type(numMyRows+1);
    for (int i=0; i<numMyRows+1; ++i) {
      row_map_size_type[i] = static_cast<size_type>(row_map[i]);
    }
    // Attach the temporary to the input RCP, to prolong its life time. Last arg=false, so we replace possibly
    // existing data without throwing.
    Teuchos::set_extra_data(row_map_size_type,"row_map as size_type",Teuchos::outArg(lop),Teuchos::POST_DESTROY,false);

    // Create unmanaged views
    DeviceLocalMatrix<ST>::row_map_type row_map_view(row_map_size_type.getRawPtr(),numMyRows+1);
    DeviceLocalMatrix<ST>::index_type   indices_view(indices,numMyNonzeros);
    DeviceLocalMatrix<ST>::values_type  values_view(values,numMyNonzeros);

    // Build the matrix.
    DeviceLocalMatrix<ST> data("Epetra device data", numMyRows, numMyCols, numMyNonzeros, values_view, row_map_view, indices_view);
    return data;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getDeviceData! Could not cast Thyra_Vector to any of the supported concrete types.\n");

  // Dummy return value, to silence compiler warnings
  DeviceLocalMatrix<const ST> dummy;
  return dummy;
}
    template<int I>
    struct ShowMeI {};

DeviceLocalMatrix<ST> getNonconstDeviceData (Teuchos::RCP<Thyra_LinearOp>& lop)
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
                                "Error in getNonconstDeviceData! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");

    // If you want the output DeviceLocalMatrix to have view semantic on the matrix values,
    // you need to use the constructor that 'views' the input arrays.
    // So we need to create views unmanaged, which need to view the matrix data.
    // If it's not possible to view the matrix data, we need to create temporaries,
    // and view those. If that's the case, we need to attach the temporaries to the
    // input RCP, so that they live as long as the input LinearOp.
    // WARNING: This is *highly* relying on Epetra_CrsMatrix internal storage.
    //          More precisely, I'm not even sure this routine could be fixed
    //          if Epetra_CrsMatrix changes the internal storage scheme.

    using StaticGraphType = DeviceLocalMatrix<ST>::staticcrsgraph_type;
    using size_type = StaticGraphType::size_type;

    // Some data from the matrix
    const int numMyRows = emat->NumMyRows();
    const int numMyCols = emat->NumMyCols();
    const int numMyNonzeros = emat->NumMyNonzeros();

    // Grab the data
    LO* row_map;
    LO* indices;
    ST* values;
    int err_code = emat->ExtractCrsDataPointers(row_map,indices,values);
    ALBANY_EXPECT(err_code==0, "Error in getNonconstDeviceData! Something went wrong while extracting Epetra_CrsMatrix local data pointers.\n");
    (void) err_code;
    Teuchos::ArrayRCP<size_type> row_map_size_type(numMyRows+1);
    for (int i=0; i<numMyRows+1; ++i) {
      row_map_size_type[i] = static_cast<size_type>(row_map[i]);
    }
    // Attach the temporary to the input RCP, to prolong its life time. Last arg=false, so we replace possibly
    // existing data without throwing.
    Teuchos::set_extra_data(row_map_size_type,"row_map as size_type",Teuchos::outArg(lop),Teuchos::POST_DESTROY,false);

    // Create unmanaged views
    DeviceLocalMatrix<ST>::row_map_type row_map_view(row_map_size_type.getRawPtr(),numMyRows+1);
    DeviceLocalMatrix<ST>::index_type   indices_view(indices,numMyNonzeros);
    DeviceLocalMatrix<ST>::values_type  values_view(values,numMyNonzeros);

    // Build the matrix.
    DeviceLocalMatrix<ST> data("Epetra device data", numMyRows, numMyCols, numMyNonzeros, values_view, row_map_view, indices_view);
    return data;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getNonconstDeviceData! Could not cast Thyra_Vector to any of the supported concrete types.\n");

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
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getNnconstLocalData! Could not cast Thyra_Vector to any of the supported concrete types.\n");
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
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getLocalData! Could not cast Thyra_Vector to any of the supported concrete types.\n");
    }
  }

  return vals;
}

int getNumVectors (const Teuchos::RCP<const Thyra_MultiVector>& mv)
{
  auto tv = getConstTpetraMultiVector(mv,false);
  if (!tv.is_null()) {
    return tv->getNumVectors(); 
  }
#if defined(ALBANY_EPETRA)
  auto ev = getConstEpetraMultiVector(mv,false);
  if (!ev.is_null()) {
    return ev->NumVectors(); 
  }
#endif 
  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getNumVectors! Could not cast Thyra_MultiVector to any of the supported concrete types.\n");
}


Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>>
getNonconstLocalData (const Teuchos::RCP<Thyra_MultiVector>& mv)
{
  if (mv.is_null()) {
    return Teuchos::null;
  }

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> data(mv->domain()->dim());
  for (int i=0; i<mv->domain()->dim(); ++i) {
    data[i] = getNonconstLocalData(mv->col(i));
  }
  return data;
}

Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>>
getLocalData (const Teuchos::RCP<const Thyra_MultiVector>& mv)
{
  if (mv.is_null()) {
    return Teuchos::null;
  }

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> data(mv->domain()->dim());
  for (int i=0; i<mv->domain()->dim(); ++i) {
    data[i] = getLocalData(mv->col(i));
  }
  return data;
}

Teuchos::ArrayRCP<ST> getNonconstLocalData (Thyra_Vector& v) {
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
    auto* spmd_v = dynamic_cast<Thyra::SpmdVectorBase<ST>*>(&v);
    if (spmd_v!=nullptr) {
      spmd_v->getNonconstLocalData(Teuchos::outArg(vals));
    } else {
      // If all the tries above are unsuccessful, throw an error.
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getNonconstLocalData! Could not cast Thyra_Vector to any of the supported concrete types.\n");
    }
  }

  return vals;
}

Teuchos::ArrayRCP<const ST> getLocalData (const Thyra_Vector& v) {
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
    auto* spmd_v = dynamic_cast<const Thyra::SpmdVectorBase<ST>*>(&v);
    if (spmd_v!=nullptr) {
      spmd_v->getLocalData(Teuchos::outArg(vals));
    } else {
      // If all the tries above are unsuccessful, throw an error.
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getLocalData! Could not cast Thyra_Vector to any of the supported concrete types.\n");
    }
  }

  return vals;
}

Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> getNonconstLocalData (Thyra_MultiVector& mv) {
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> data(mv.domain()->dim());
  for (int i=0; i<mv.domain()->dim(); ++i) {
    data[i] = getNonconstLocalData(mv.col(i));
  }
  return data;
}

Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> getLocalData (const Thyra_MultiVector& mv)
{
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>> data(mv.domain()->dim());
  for (int i=0; i<mv.domain()->dim(); ++i) {
    data[i] = getLocalData(mv.col(i));
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
                                "Error in getDeviceData! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");
    DeviceView1d<const ST> data( evec->Values(), evec->MyLength() );
    return data;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getDeviceData! Could not cast Thyra_Vector to any of the supported concrete types.\n");

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
                                "Error in getNonconstDeviceData! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");
    DeviceView1d<ST> data( evec->Values(), evec->MyLength() );
    return data;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in getNonconstDeviceData! Could not cast Thyra_Vector to any of the supported concrete types.\n");

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
void describe<Thyra_VectorSpace> (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                                  Teuchos::FancyOStream& out,
                                  const Teuchos::EVerbosityLevel verbLevel)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tvs = getTpetraMap(vs,false);
  if (!tvs.is_null()) {
    tvs->describe(out,verbLevel);
    return;
  }

#if defined(ALBANY_EPETRA)
  auto evs = getEpetraBlockMap(vs,false);
  if (!evs.is_null()) {
    evs->Print(*out.getOStream());
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in describe! Could not cast Thyra_Vector to any of the supported concrete types.\n");
}


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
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in describe! Could not cast Thyra_Vector to any of the supported concrete types.\n");
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
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in describe! Could not cast Thyra_Vector to any of the supported concrete types.\n");
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
    //map needed to determine numbering of the rhs
    writeMatrixMarket(tv->getMap(),prefix+"_map",counter);
    return;
  }

#if defined(ALBANY_EPETRA)
  auto ev = getConstEpetraVector(v,false);
  if (!ev.is_null()) {
    // TODO: avoid petra conversion, and call EpetraExt I/O directly
    tv = Petra::EpetraVector_To_TpetraVectorConst(*ev,createTeuchosCommFromEpetraComm(ev->Comm()));
    writeMatrixMarket(tv,prefix,counter);
    //map needed to determine numbering of the rhs
    writeMatrixMarket(tv->getMap(),prefix+"_map",counter);
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in writeMatrixMarket! Could not cast Thyra_Vector to any of the supported concrete types.\n");
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
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in writeMatrixMarket! Could not cast Thyra_Vector to any of the supported concrete types.\n");
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
    return;
  } else {
    // It may be a Thyra::EpetraLinearOp. Try to extract the op
    auto eLop = Teuchos::rcp_dynamic_cast<const Thyra::EpetraLinearOp>(A);
    if (!eLop.is_null()) {
      auto eMat = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(eLop->epetra_op());
      TEUCHOS_TEST_FOR_EXCEPTION (eMat.is_null(), std::logic_error,
                                  "Error in writeMatrixMarket! The thyra linear op is of type Thyra::EpetraLinearOp, "
                                  "but the stored Epetra_Operator rcp is either null or not of concrete type Epetra_CrsMatrix.\n");
      tA = Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(*eMat,createTeuchosCommFromEpetraComm(eMat->Comm()));
      writeMatrixMarket(tA,prefix,counter);
      return;
    }
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in writeMatrixMarket! Could not cast Thyra_LinearOp to any of the supported concrete types.\n");
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
    return;
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error in writeMatrixMarket! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");
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
