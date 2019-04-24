#include "Albany_CombineAndScatterManagerEpetra.hpp"

#include "Albany_EpetraThyraUtils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Macros.hpp"

namespace {
Epetra_CombineMode combineModeE (const Albany::CombineMode modeA)
{
  Epetra_CombineMode modeE;
  switch (modeA) {
    case Albany::CombineMode::ADD:
      modeE = Epetra_CombineMode::Add;
      break;
    case Albany::CombineMode::INSERT:
      modeE = Epetra_CombineMode::Insert;
      break;
    case Albany::CombineMode::ZERO:
      modeE = Epetra_CombineMode::Zero;
      break;
    case Albany::CombineMode::ABSMAX:
      modeE = Epetra_CombineMode::AbsMax;
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Unknown Albany combine mode. Please, contact developers.\n");
  }
  return modeE;
}

} // anonymous namespace

namespace Albany
{

CombineAndScatterManagerEpetra::
CombineAndScatterManagerEpetra(const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                               const Teuchos::RCP<const Thyra_VectorSpace>& overlapped)
 : CombineAndScatterManager(owned,overlapped)
{
  auto ownedE = getEpetraMap(owned);
  auto overlappedE = getEpetraMap(overlapped);

  importer = Teuchos::rcp( new Epetra_Import(*overlappedE, *ownedE) );
}

void CombineAndScatterManagerEpetra::
combine (const Thyra_Vector& src,
               Thyra_Vector& dst,
         const CombineMode CM) const
{
  auto cmE = combineModeE(CM);
  auto srcE = getConstEpetraVector(src);
  auto dstE = getEpetraVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input src vector does not match the importer's target map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input dst vector does not match the importer's source map.\n");
#endif

  dstE->Export(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
combine (const Thyra_MultiVector& src,
               Thyra_MultiVector& dst,
         const CombineMode CM) const
{
  // There's a catch here!
  // Legend: V = Vector, MV = MultiVector, TV = Epetra_Vector, TMV = Epetra_MultiVector, T_xyz = Thyra_xyz
  // One can create a T_TV, then pass it to routines expecting a T_MV, since T_TV inherits from T_V,
  // which inherits from T_MV. However, T_TV does NOT inherit from T_TMV, so such routines would
  // try to cast the input to T_TMV and fail. This would be solved if T_TV also inherited
  // from T_TMV, but that's hard to do (without code duplication), since T_T(M)V store
  // ConstNonConstObj containers to the Epetra objects, which I _think_ do not support polymorphism.
  // So, given what we have, we _try_ to extract a TMV from the T_MV, and, if we fail,
  // we try again, this time extracting a TV. If we still fail, then we can error out.
  Teuchos::RCP<const Epetra_MultiVector> srcE = getConstEpetraMultiVector(src,false);
  Teuchos::RCP<Epetra_MultiVector> dstE = getEpetraMultiVector(dst,false);

  if (srcE.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Epetra_Vector
    const Thyra_Vector* srcV = dynamic_cast<const Thyra_Vector*>(&src);

    TEUCHOS_TEST_FOR_EXCEPTION (srcV==nullptr, std::runtime_error,
                                "Error! Input src does not seem to be a Epetra_MultiVector or a Epetra_Vector.\n");

    // This time throw if extraction fails
    srcE = getConstEpetraVector(*srcV);
  }

  if (dstE.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Epetra_Vector
    Thyra_Vector* dstV = dynamic_cast<Thyra_Vector*>(&dst);

    TEUCHOS_TEST_FOR_EXCEPTION (dstV==nullptr, std::runtime_error,
                                "Error! Input dst does not seem to be a Epetra_MultiVector or a Epetra_Vector.\n");

    // This time throw if extraction fails
    dstE = getEpetraVector(*dstV);
  }

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's target map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's source map.\n");
#endif

  auto cmE = combineModeE(CM);
  dstE->Export(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
combine (const Thyra_LinearOp& src,
               Thyra_LinearOp& dst,
         const CombineMode CM) const
{
  auto cmE = combineModeE(CM);
  auto srcE = getConstEpetraMatrix(src);
  auto dstE = getEpetraMatrix(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The row map of the input src matrix does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The row map of the input dst matrix does not match the importer's target map.\n");
#endif

  dstE->Export(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
combine (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<      Thyra_Vector>& dst,
         const CombineMode CM) const
{
  auto cmE = combineModeE(CM);
  auto srcE = getConstEpetraVector(src);
  auto dstE = getEpetraVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input src vector does not match the importer's target map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input dst vector does not match the importer's source map.\n");
#endif

  dstE->Export(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
         const Teuchos::RCP<      Thyra_MultiVector>& dst,
         const CombineMode CM) const
{
  // There's a catch here!
  // Legend: V = Vector, MV = MultiVector, TV = Epetra_Vector, TMV = Epetra_MultiVector, T_xyz = Thyra_xyz
  // One can create a T_TV, then pass it to routines expecting a T_MV, since T_TV inherits from T_V,
  // which inherits from T_MV. However, T_TV does NOT inherit from T_TMV, so such routines would
  // try to cast the input to T_TMV and fail. This would be solved if T_TV also inherited
  // from T_TMV, but that's hard to do (without code duplication), since T_T(M)V store
  // ConstNonConstObj containers to the Epetra objects, which I _think_ do not support polymorphism.
  // So, given what we have, we _try_ to extract a TMV from the T_MV, and, if we fail,
  // we try again, this time extracting a TV. If we still fail, then we can error out.
  Teuchos::RCP<const Epetra_MultiVector> srcE = getConstEpetraMultiVector(src,false);
  Teuchos::RCP<Epetra_MultiVector> dstE = getEpetraMultiVector(dst,false);

  if (srcE.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Epetra_Vector
    Teuchos::RCP<const Thyra_Vector> srcV = Teuchos::rcp_dynamic_cast<const Thyra_Vector>(src);

    TEUCHOS_TEST_FOR_EXCEPTION (srcV.is_null(), std::runtime_error,
                                "Error! Input src does not seem to be a Epetra_MultiVector or a Epetra_Vector.\n");

    // This time throw if extraction fails
    srcE = getConstEpetraVector(srcV);
  }

  if (dstE.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Epetra_Vector
    Teuchos::RCP<Thyra_Vector> dstV = Teuchos::rcp_dynamic_cast<Thyra_Vector>(dst);

    TEUCHOS_TEST_FOR_EXCEPTION (dstV.is_null(), std::runtime_error,
                                "Error! Input dst does not seem to be a Epetra_MultiVector or a Epetra_Vector.\n");

    // This time throw if extraction fails
    dstE = getEpetraVector(dstV);
  }

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's target map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's source map.\n");
#endif

  auto cmE = combineModeE(CM);
  dstE->Export(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
         const Teuchos::RCP<      Thyra_LinearOp>& dst,
         const CombineMode CM) const
{
  auto cmE = combineModeE(CM);
  auto srcE = getConstEpetraMatrix(src);
  auto dstE = getEpetraMatrix(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The row map of the input src matrix does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The row map of the input dst matrix does not match the importer's target map.\n");
#endif

  dstE->Export(*srcE,*importer,cmE);
}

// Scatter methods
void CombineAndScatterManagerEpetra::
scatter (const Thyra_Vector& src,
               Thyra_Vector& dst,
         const CombineMode CM) const
{
  auto cmE = combineModeE(CM);
  auto srcE = getConstEpetraVector(src);
  auto dstE = getEpetraVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input src vector does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input dst vector does not match the importer's target map.\n");
#endif

  dstE->Import(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
scatter (const Thyra_MultiVector& src,
               Thyra_MultiVector& dst,
         const CombineMode CM) const
{
  // There's a catch here!
  // Legend: V = Vector, MV = MultiVector, TV = Epetra_Vector, TMV = Epetra_MultiVector, T_xyz = Thyra_xyz
  // One can create a T_TV, then pass it to routines expecting a T_MV, since T_TV inherits from T_V,
  // which inherits from T_MV. However, T_TV does NOT inherit from T_TMV, so such routines would
  // try to cast the input to T_TMV and fail. This would be solved if T_TV also inherited
  // from T_TMV, but that's hard to do (without code duplication), since T_T(M)V store
  // ConstNonConstObj containers to the Epetra objects, which I _think_ do not support polymorphism.
  // So, given what we have, we _try_ to extract a TMV from the T_MV, and, if we fail,
  // we try again, this time extracting a TV. If we still fail, then we can error out.
  Teuchos::RCP<const Epetra_MultiVector> srcE = getConstEpetraMultiVector(src,false);
  Teuchos::RCP<Epetra_MultiVector> dstE = getEpetraMultiVector(dst,false);

  if (srcE.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Epetra_Vector
    const Thyra_Vector* srcV = dynamic_cast<const Thyra_Vector*>(&src);

    TEUCHOS_TEST_FOR_EXCEPTION (srcV==nullptr, std::runtime_error,
                                "Error! Input src does not seem to be a Epetra_MultiVector or a Epetra_Vector.\n");

    // This time throw if extraction fails
    srcE = getConstEpetraVector(*srcV);
  }

  if (dstE.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Epetra_Vector
    Thyra_Vector* dstV = dynamic_cast<Thyra_Vector*>(&dst);

    TEUCHOS_TEST_FOR_EXCEPTION (dstV==nullptr, std::runtime_error,
                                "Error! Input dst does not seem to be a Epetra_MultiVector or a Epetra_Vector.\n");

    // This time throw if extraction fails
    dstE = getEpetraVector(*dstV);
  }

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's target map.\n");
#endif

  auto cmE = combineModeE(CM);
  dstE->Import(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
scatter (const Thyra_LinearOp& src,
               Thyra_LinearOp& dst,
         const CombineMode CM) const
{
  auto cmE  = combineModeE(CM);
  auto srcE = getConstEpetraMatrix(src);
  auto dstE = getEpetraMatrix(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The row map of the input src matrix does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The row map of the input dst matrix does not match the importer's target map.\n");
#endif

  dstE->Import(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
scatter (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<      Thyra_Vector>& dst,
         const CombineMode CM) const
{
  auto cmE = combineModeE(CM);
  auto srcE = getConstEpetraVector(src);
  auto dstE = getEpetraVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input src vector does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input dst vector does not match the importer's target map.\n");
#endif

  dstE->Import(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
         const Teuchos::RCP<      Thyra_MultiVector>& dst,
         const CombineMode CM) const
{
  // There's a catch here!
  // Legend: V = Vector, MV = MultiVector, TV = Epetra_Vector, TMV = Epetra_MultiVector, T_xyz = Thyra_xyz
  // One can create a T_TV, then pass it to routines expecting a T_MV, since T_TV inherits from T_V,
  // which inherits from T_MV. However, T_TV does NOT inherit from T_TMV, so such routines would
  // try to cast the input to T_TMV and fail. This would be solved if T_TV also inherited
  // from T_TMV, but that's hard to do (without code duplication), since T_T(M)V store
  // ConstNonConstObj containers to the Epetra objects, which I _think_ do not support polymorphism.
  // So, given what we have, we _try_ to extract a TMV from the T_MV, and, if we fail,
  // we try again, this time extracting a TV. If we still fail, then we can error out.
  Teuchos::RCP<const Epetra_MultiVector> srcE = getConstEpetraMultiVector(src,false);
  Teuchos::RCP<Epetra_MultiVector> dstE = getEpetraMultiVector(dst,false);

  if (srcE.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Epetra_Vector
    Teuchos::RCP<const Thyra_Vector> srcV = Teuchos::rcp_dynamic_cast<const Thyra_Vector>(src);

    TEUCHOS_TEST_FOR_EXCEPTION (srcV.is_null(), std::runtime_error,
                                "Error! Input src does not seem to be a Epetra_MultiVector or a Epetra_Vector.\n");

    // This time throw if extraction fails
    srcE = getConstEpetraVector(srcV);
  }

  if (dstE.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Epetra_Vector
    Teuchos::RCP<Thyra_Vector> dstV = Teuchos::rcp_dynamic_cast<Thyra_Vector>(dst);

    TEUCHOS_TEST_FOR_EXCEPTION (dstV.is_null(), std::runtime_error,
                                "Error! Input dst does not seem to be a Epetra_MultiVector or a Epetra_Vector.\n");

    // This time throw if extraction fails
    dstE = getEpetraVector(dstV);
  }

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's target map.\n");
#endif

  auto cmE = combineModeE(CM);
  dstE->Import(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
         const Teuchos::RCP<      Thyra_LinearOp>& dst,
         const CombineMode CM) const
{
  auto cmE  = combineModeE(CM);
  auto srcE = getConstEpetraMatrix(src);
  auto dstE = getEpetraMatrix(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcE->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The row map of the input src matrix does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstE->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The row map of the input dst matrix does not match the importer's target map.\n");
#endif

  dstE->Import(*srcE,*importer,cmE);
}

void CombineAndScatterManagerEpetra::
create_ghosted_aura_owners () const {
  // Use the getter, so it creates the vs is if it's null
  auto gvs = getGhostedAuraVectorSpace();

  // Get the gids of the ghosted vs
  auto gids = getGlobalElements(gvs);
  Teuchos::Array<Epetra_GO> egids_array;
  Teuchos::ArrayView<Epetra_GO> egids;
  if (sizeof(GO)==sizeof(Epetra_GO)) {
    // Same size, potentially different type name. A reinterpret_cast will do.
    egids = Teuchos::arrayView(reinterpret_cast<Epetra_GO*>(gids.getRawPtr()),gids.size());
  } else {
    // Cannot reinterpret cast. Need to copy gids into Epetra_GO array
    egids_array.resize(gids.size());
    const GO max_safe_gid = static_cast<GO>(Teuchos::OrdinalTraits<Epetra_GO>::max());
    for (int i=0; i<gids.size(); ++i) {
      ALBANY_EXPECT(gids[i]<=max_safe_gid, "Error in createLocallyReplicatedVectorSpace! Input gids exceed Epetra_GO ranges.\n");
      egids_array[i] = static_cast<Epetra_GO>(gids[i]);
    }
    (void) max_safe_gid;
    egids = Teuchos::arrayView(egids_array.getRawPtr(),gids.size());
  }

  // Ask epetra to give the pid of the owner of each of the gids
  Teuchos::Array<LO> lids(gids.size());
  ghosted_aura_owners.resize(lids.size());
  auto map = getEpetraMap(getOwnedAuraVectorSpace());
  map->RemoteIDList(gids.size(),egids.getRawPtr(),ghosted_aura_owners.getRawPtr(),lids.getRawPtr());
}

void CombineAndScatterManagerEpetra::
create_owned_aura_users () const {
  // Use the getter, so it creates the vs is if it's null
  auto ga_vs = getGhostedAuraVectorSpace();
  auto oa_vs = getOwnedAuraVectorSpace();
  auto ga_map = getEpetraMap(ga_vs);
  auto oa_map = getEpetraMap(oa_vs);

  // Build an importer from the owned to the ghosted
  auto imp = Teuchos::rcp( new Epetra_Import(*ga_map, *oa_map) );

  // Get the pid to which each of the exported pids goes
  auto pids = imp->ExportPIDs();
  auto lids = imp->ExportLIDs();
  for (int i=0; i<imp->NumExportIDs(); ++i) {
    owned_aura_users.push_back(std::make_pair(lids[i],pids[i]));
  }
}

} // namespace Albany
