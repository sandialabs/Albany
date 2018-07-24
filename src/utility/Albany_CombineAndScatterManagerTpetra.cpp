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
{
  auto ownedT = Albany::getTpetraMap(owned);
  auto overlappedT = Albany::getTpetraMap(overlapped);

  TEUCHOS_TEST_FOR_EXCEPTION(ownedT.is_null(), std::runtime_error, "Error! Could not cast owned vector space to Tpetra type.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(overlappedT.is_null(), std::runtime_error, "Error! Could not cast overlapped vector space to Tpetra type.\n");

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

  TEUCHOS_TEST_FOR_EXCEPTION(srcT.is_null(), std::runtime_error, "Error! Could not cast src vector to Tpetra type.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(dstT.is_null(), std::runtime_error, "Error! Could not cast dst vector to Tpetra type.\n");
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
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraMultiVector(src);
  auto dstT = Albany::getTpetraMultiVector(dst);

  TEUCHOS_TEST_FOR_EXCEPTION(srcT.is_null(), std::runtime_error, "Error! Could not cast src multi vector to Tpetra type.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(dstT.is_null(), std::runtime_error, "Error! Could not cast dst multi vector to Tpetra type.\n");
#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's target map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's source map.\n");
#endif

  dstT->doExport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
combine (const Teuchos::RCP<const Thyra_LinearOp>& /* src */,
         const Teuchos::RCP<Thyra_LinearOp>& /* dst */,
         const CombineMode /* CM */) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Combine for linear operators not implemented yet.\n");
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

  TEUCHOS_TEST_FOR_EXCEPTION(srcT.is_null(), std::runtime_error, "Error! Could not cast src vector to Tpetra type.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(dstT.is_null(), std::runtime_error, "Error! Could not cast dst vector to Tpetra type.\n");
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
  auto cmT = combineModeT(CM);
  auto srcT = Albany::getConstTpetraMultiVector(src);
  auto dstT = Albany::getTpetraMultiVector(dst);

  TEUCHOS_TEST_FOR_EXCEPTION(srcT.is_null(), std::runtime_error, "Error! Could not cast src multi vector to Tpetra type.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(dstT.is_null(), std::runtime_error, "Error! Could not cast dst multi vector to Tpetra type.\n");
#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION(!srcT->getMap()->isSameAs(*importer->getSourceMap()), std::runtime_error,
                             "Error! The map of the input src multi vector does not match the importer's source map.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(!dstT->getMap()->isSameAs(*importer->getTargetMap()), std::runtime_error,
                             "Error! The map of the input dst multi vector does not match the importer's target map.\n");
#endif

  dstT->doImport(*srcT,*importer,cmT);
}

void CombineAndScatterManagerTpetra::
scatter (const Teuchos::RCP<const Thyra_LinearOp>& /* src */,
              const Teuchos::RCP<Thyra_LinearOp>& /* dst */,
              const CombineMode /* CM */) const
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Scatter for linear operators not implemented yet.\n");
}

} // namespace Albany
