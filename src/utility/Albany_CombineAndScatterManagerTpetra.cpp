#include "Albany_CombineAndScatterManagerTpetra.hpp"

#include "Albany_TpetraThyraTypes.hpp"
#include "Albany_TpetraThyraUtils.hpp"

namespace {
Tpetra::CombineMode combineModeT (const Albany::CombineMode modeA)
{
  Tpetra::CombineMode modeT;
  switch (modeA) {
    case Albany::CombineMode::ADD:
      modeT = Tpetra::CombineMode::ADD;
      break;
    case Albany::CombineMode::INSERT:
      modeT = Tpetra::CombineMode::INSERT;
      break;
    case Albany::CombineMode::ZERO:
      modeT = Tpetra::CombineMode::ZERO;
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error! Unknown Albany combine mode. Please, contact developers.\n");
  }
  return modeT;
}

} // anonymous namespace

namespace Albany
{

CombineAndScatterManagerTpetra::
CombineAndScatterManagerTpetra(const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                               const Teuchos::RCP<const Thyra_VectorSpace>& overlapped)
 : owned_vs      (owned)
 , overlapped_vs (overlapped)
{
  auto ownedT = Albany::getTpetraMap(owned);
  auto overlappedT = Albany::getTpetraMap(overlapped);

  importer = Teuchos::rcp( new Tpetra_Import(ownedT, overlappedT) );
}

void CombineAndScatterManagerTpetra::
combine (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<Thyra_Vector>& dst,
         const CombineMode CM) const
{
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraVector(src);
  auto dstT = Albany::getTpetraVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input src vector does not match the importer's target map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input dst vector does not match the importer's source map.\n");
#endif

  dstT->doExport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
         const Teuchos::RCP<Thyra_MultiVector>& dst,
         const CombineMode CM) const
{
  // There's a catch here!
  // Legend: V = Vector, MV = MultiVector, TV = Tpetra_Vector, TMV = Tpetra_MultiVector, T_xyz = Thyra_xyz
  // One can create a T_TV, then pass it to routines expecting a T_MV, since T_TV inherits from T_V,
  // which inherits from T_MV. However, T_TV does NOT inherit from T_TMV, so such routines would
  // try to cast the input to T_TMV and fail. This would be solved if T_TV also inherited
  // from T_TMV, but that's hard to do (without code duplication), since T_T(M)V store
  // ConstNonConstObj containers to the Tpetra objects, which I _think_ do not support polymorphism.
  // So, given what we have, we _try_ to extract a TMV from the T_MV, and, if we fail,
  // we try again, this time extracting a TV. If we still fail, then we can error out.
  Teuchos::RCP<const Tpetra_MultiVector> srcT = Albany::getConstTpetraMultiVector(src,false);
  Teuchos::RCP<Tpetra_MultiVector> dstT = Albany::getTpetraMultiVector(dst,false);

  if (srcT.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Tpetra_Vector
    auto srcV = Teuchos::rcp_dynamic_cast<const Thyra_Vector>(src);

    TEUCHOS_TEST_FOR_EXCEPTION (srcV.is_null(), std::runtime_error,
                                "Error! Input src does not seem to be a TpetraMultiVector or a Thyra_Vector.\n");

    // This time throw if extraction fails
    srcT = Albany::getConstTpetraVector(srcV);
  }

  if (dstT.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Tpetra_Vector
    auto dstV = Teuchos::rcp_dynamic_cast<Thyra_Vector>(dst);

    TEUCHOS_TEST_FOR_EXCEPTION (dstV.is_null(), std::runtime_error,
                                "Error! Input dst does not seem to be a TpetraMultiVector or a Thyra_Vector.\n");

    // This time throw if extraction fails
    dstT = Albany::getTpetraVector(dstV);
  }

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's target map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's source map.\n");
#endif

  auto cmT = combineModeT(CM);
  dstT->doExport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
         const Teuchos::RCP<Thyra_LinearOp>& dst,
         const CombineMode CM) const
{
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraMatrix(src);
  auto dstT = Albany::getTpetraMatrix(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The row map of the input src matrix does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The row map of the input dst matrix does not match the importer's target map.\n");
#endif

  dstT->doExport(*srcT,*importer,cmT);
}

// Scatter methods
void CombineAndScatterManagerTpetra::
scatter (const Teuchos::RCP<const Thyra_Vector>& src,
         const Teuchos::RCP<Thyra_Vector>& dst,
         const CombineMode CM) const
{
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraVector(src);
  auto dstT = Albany::getTpetraVector(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input src vector does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input dst vector does not match the importer's target map.\n");
#endif

  dstT->doImport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
         const Teuchos::RCP<Thyra_MultiVector>& dst,
         const CombineMode CM) const
{
  // There's a catch here!
  // Legend: V = Vector, MV = MultiVector, TV = Tpetra_Vector, TMV = Tpetra_MultiVector, T_xyz = Thyra_xyz
  // One can create a T_TV, then pass it to routines expecting a T_MV, since T_TV inherits from T_V,
  // which inherits from T_MV. However, T_TV does NOT inherit from T_TMV, so such routines would
  // try to cast the input to T_TMV and fail. This would be solved if T_TV also inherited
  // from T_TMV, but that's hard to do (without code duplication), since T_T(M)V store
  // ConstNonConstObj containers to the Tpetra objects, which I _think_ do not support polymorphism.
  // So, given what we have, we _try_ to extract a TMV from the T_MV, and, if we fail,
  // we try again, this time extracting a TV. If we still fail, then we can error out.
  Teuchos::RCP<const Tpetra_MultiVector> srcT = Albany::getConstTpetraMultiVector(src,false);
  Teuchos::RCP<Tpetra_MultiVector> dstT = Albany::getTpetraMultiVector(dst,false);

  if (srcT.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Tpetra_Vector
    auto srcV = Teuchos::rcp_dynamic_cast<const Thyra_Vector>(src);

    TEUCHOS_TEST_FOR_EXCEPTION (srcV.is_null(), std::runtime_error,
                                "Error! Input src does not seem to be a TpetraMultiVector or a Thyra_Vector.\n");

    // This time throw if extraction fails
    srcT = Albany::getConstTpetraVector(srcV);
  }

  if (dstT.is_null()) {
    // Try to cast to Thyra_Vector, then extract the Tpetra_Vector
    auto dstV = Teuchos::rcp_dynamic_cast<Thyra_Vector>(dst);

    TEUCHOS_TEST_FOR_EXCEPTION (dstV.is_null(), std::runtime_error,
                                "Error! Input dst does not seem to be a TpetraMultiVector or a Thyra_Vector.\n");

    // This time throw if extraction fails
    dstT = Albany::getTpetraVector(dstV);
  }

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's target map.\n");
#endif

  auto cmT = combineModeT(CM);
  dstT->doImport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
         const Teuchos::RCP<Thyra_LinearOp>& dst,
         const CombineMode CM) const
{
  auto cmT  = combineModeT(CM);
  auto srcT = Albany::getConstTpetraMatrix(src);
  auto dstT = Albany::getTpetraMatrix(dst);

#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The row map of the input src matrix does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The row map of the input dst matrix does not match the importer's target map.\n");
#endif

  dstT->doImport(*srcT,*importer,cmT);
}

} // namespace Albany
