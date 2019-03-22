#include "Albany_ThyraUtils.hpp"
#include "Albany_CommUtils.hpp"
#include "Albany_ThyraCrsMatrixFactory.hpp"
#include "Albany_TpetraThyraUtils.hpp"
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

Teuchos::RCP<const Thyra_VectorSpace>
createLocallyReplicatedVectorSpace (const Teuchos::ArrayView<const GO>& gids, const Teuchos::RCP<const Teuchos_Comm> comm)
{
  auto bt = build_type();
  switch (bt) {
#ifdef ALBANY_EPETRA
    case BuildType::Epetra:
    {
      Teuchos::RCP<const Epetra_BlockMap> emap;
      if (sizeof(GO)==sizeof(Epetra_GO)) {
        // Same size, potentially different type name. A reinterpret_cast will do.
        emap = Teuchos::rcp( new Epetra_BlockMap(gids.size(),gids.size(),
                             reinterpret_cast<const Epetra_GO*>(gids.getRawPtr()),
                             1,0,*createEpetraCommFromTeuchosComm(comm)) );
      } else {
        // Cannot reinterpret cast. Need to copy gids into Epetra_GO array
        Teuchos::Array<Epetra_GO> e_gids(gids.size());
        const GO max_safe_gid = static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max());
        for (int i=0; i<gids.size(); ++i) {
          ALBANY_EXPECT(gids[i]<=max_safe_gid, "Error! Input gids exceed Epetra_GO ranges.\n");
          e_gids[i] = static_cast<Epetra_GO>(gids[i]);
        }
        (void) max_safe_gid;
        emap = Teuchos::rcp( new Epetra_BlockMap(gids.size(),gids.size(),
                             reinterpret_cast<const Epetra_GO*>(e_gids.getRawPtr()),
                             1,0,*createEpetraCommFromTeuchosComm(comm)) );
      }
      return createThyraVectorSpace(emap);
      break;
    }
#endif
    case BuildType::Tpetra:
    {
      Teuchos::ArrayView<const Tpetra_GO> tgids(reinterpret_cast<const Tpetra_GO*>(gids.getRawPtr()),gids.size());
      Teuchos::RCP<const Tpetra_Map> tmap( new Tpetra_Map(tgids.size(),tgids,0,comm) );
      return createThyraVectorSpace(tmap);
      break;
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Build type not supported.\n");
  }

  TEUCHOS_UNREACHABLE_RETURN (Teuchos::null);
}

GO getGlobalElement (const Teuchos::RCP<const Thyra_VectorSpace>& vs, const LO lid) {
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs,false);
  if (!tmap.is_null()) {
    return static_cast<GO>(tmap->getGlobalElement(lid));
  }

#if defined(ALBANY_EPETRA)
  auto emap = getEpetraBlockMap(vs,false);
  if (!emap.is_null()) {
    return static_cast<GO>(emap->GID(lid));
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
    return tmap->getLocalElement(static_cast<Tpetra_GO>(gid));
  }

#if defined(ALBANY_EPETRA)
  auto emap = getEpetraBlockMap(vs,false);
  if (!emap.is_null()) {
    // Note: simply calling LID(gid) can be ambiguous, if GO!=int and GO!=long long.
    //       Hence, we explicitly cast to whatever has size 64 bits (should *always* be long long, but the if is compile time, so no penalty)
    const GO max_safe_gid = static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max());
    ALBANY_EXPECT(gid<=max_safe_gid, "Error! Input gid exceed Epetra_GO ranges.\n");
    (void) max_safe_gid;
    return emap->LID(static_cast<Epetra_GO>(gid));
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(-1);
}

void getGlobalElements (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
                        Teuchos::Array<GO>& gids)
{
  auto spmd_vs = getSpmdVectorSpace(vs);
  const LO localDim = spmd_vs->localSubDim();
  gids.resize(localDim);
  for (LO i=0; i<localDim; ++i) {
    gids[i] = getGlobalElement(vs,i);
  }
}

Teuchos::Array<GO> getGlobalElements (const Teuchos::RCP<const Thyra_VectorSpace>& vs)
{
  Teuchos::Array<GO> gids;
  getGlobalElements(vs,gids);
  return gids;
}

bool locallyOwnedComponent (const Teuchos::RCP<const Thyra_SpmdVectorSpace>& vs, const GO gid)
{
  // Allow failure, since we don't know what the underlying linear algebra is
  auto tmap = getTpetraMap(vs,false);
  if (!tmap.is_null()) {
    return tmap->isNodeGlobalElement(static_cast<Tpetra_GO>(gid));
  }

#if defined(ALBANY_EPETRA)
  auto emap = getEpetraBlockMap(vs,false);
  if (!emap.is_null()) {
    // Note: simply calling LID(gid) can be ambiguous, if GO!=int and GO!=long long.
    //       Hence, we explicitly cast to whatever has size 64 bits (should *always* be long long, but the if is compile time, so no penalty)
    const GO max_safe_gid = static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max());
    ALBANY_EXPECT(gid<=max_safe_gid, "Error! Input gid exceed Epetra_GO ranges.\n");
    (void) max_safe_gid;
    return emap->MyGID(static_cast<Epetra_GO>(gid));
  }
#endif

  // If all the tries above are unsuccessful, throw an error.
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

  // Silence compiler warning
  TEUCHOS_UNREACHABLE_RETURN(false);
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_VectorSpace to any of the supported concrete types.\n");

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
        ALBANY_EXPECT(gids[i]<=max_safe_gid, "Error! Input gids exceed Epetra_GO ranges.\n");
        egids[i] = static_cast<Epetra_GO>(gids[i]);
      }
      (void) max_safe_gid;
      emap = Teuchos::rcp( new Epetra_BlockMap(numGlobalElements,gids.size(),egids.getRawPtr(),1,0,*ecomm) );
    }
    return createThyraVectorSpace(emap);
#else
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Epetra build not supported.\n");
#endif
  } else if (bt == BuildType::Tpetra) {
    auto gsi = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const decltype(gsi) numGlobalElements = (globalDim==invalid) ? gsi : static_cast<Epetra_GO>(globalDim);
    Teuchos::ArrayView<const Tpetra_GO> tgids(reinterpret_cast<const Tpetra_GO*>(gids.getRawPtr()),gids.size());
    Teuchos::RCP<const Tpetra_Map> tmap = Teuchos::rcp( new Tpetra_Map(numGlobalElements,tgids,0,comm) );
    return createThyraVectorSpace(tmap);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Invalid or unsupported build type.\n");
  }
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
    emat->OptimizeStorage();  // This allows to extract data with 'ExtractCrsDataPointers
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

Teuchos::RCP<const Thyra_LinearOp>
buildRestrictionOperator (const Teuchos::RCP<const Thyra_VectorSpace>& space,
                          const Teuchos::RCP<const Thyra_VectorSpace>& subspace)
{
  // In the process, verify the that subspace is a subspace of space
  auto spmd_space    = getSpmdVectorSpace(space);
  auto spmd_subspace = getSpmdVectorSpace(subspace);

  ThyraCrsMatrixFactory factory(space,subspace,1,true);

  const int localSubDim = spmd_subspace->localSubDim();
  for (LO lid=0; lid<localSubDim; ++lid) {
    const GO gid = getGlobalElement(spmd_subspace,lid);
    TEUCHOS_TEST_FOR_EXCEPTION (!locallyOwnedComponent(spmd_space,gid), std::logic_error,
                                "Error! The input 'subspace' is not a subspace of the input 'space'.\n");
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
  auto spmd_space    = getSpmdVectorSpace(space);
  auto spmd_subspace = getSpmdVectorSpace(subspace);

  ThyraCrsMatrixFactory factory(subspace,space,1,true);

  const int localSubDim = spmd_subspace->localSubDim();
  for (LO lid=0; lid<localSubDim; ++lid) {
    const GO gid = getGlobalElement(spmd_subspace,lid);
    TEUCHOS_TEST_FOR_EXCEPTION (!locallyOwnedComponent(spmd_space,gid), std::logic_error,
                                "Error! The input 'subspace' is not a subspace of the input 'space'.\n");
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
                                "Error! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");

    // If you want the output DeviceLocalMatrix to have view semantic on the matrix values,
    // you need to use the constructor that 'views' the input arrays.
    // So we need to create views unmanaged, which need to view the matrix data.
    // WARNING: This is *highly* relying on Epetra_CrsMatrix internal storage.
    //          More precisely, I'm not even sure this routine could be fixed
    //          if Epetra_CrsMatrix changes the internal storage scheme.

    using StaticGraphType = DeviceLocalMatrix<ST>::staticcrsgraph_type;
    using size_type = StaticGraphType::size_type;
    TEUCHOS_TEST_FOR_EXCEPTION (sizeof(size_type)!=sizeof(LO), std::runtime_error,
                                "Error! Extracting local data from an Epetra_CrsMatrix is safe only as long as "
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
    ALBANY_EXPECT(err_code==0, "Error! Something went wrong while extracting Epetra_CrsMatrix local data pointers.\n");
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");

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
                                "Error! Cannot use Epetra if the memory space of PHX::Device is not the HostSpace.\n");

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
    ALBANY_EXPECT(err_code==0, "Error! Something went wrong while extracting Epetra_CrsMatrix local data pointers.\n");
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
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
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
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
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
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Could not cast Thyra_Vector to any of the supported concrete types.\n");
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
    return;
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
    return;
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
    return;
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
    return;
  } else {
    // It may be a Thyra::EpetraLinearOp. Try to extract the op
    auto eLop = Teuchos::rcp_dynamic_cast<const Thyra::EpetraLinearOp>(A);
    if (!eLop.is_null()) {
      auto eMat = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(eLop->epetra_op());
      TEUCHOS_TEST_FOR_EXCEPTION (eMat.is_null(), std::logic_error,
                                  "Error! The thyra linear op is of type Thyra::EpetraLinearOp, "
                                  "but the stored Epetra_Operator rcp is either null or not of concrete type Epetra_CrsMatrix.\n");
      tA = Petra::EpetraCrsMatrix_To_TpetraCrsMatrix(*eMat,createTeuchosCommFromEpetraComm(eMat->Comm()));
      writeMatrixMarket(tA,prefix,counter);
      return;
    }
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
    return;
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
