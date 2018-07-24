#include "Albany_TpetraThyraUtils.hpp"

#include "Albany_TpetraThyraTypes.hpp"

namespace Albany
{

// ============ Tpetra->Thyra conversion routines ============ //

Teuchos::RCP<Thyra_Vector> 
createThyraVector (const Teuchos::RCP<Tpetra_Vector> v)
{
  Teuchos::RCP<Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    v_thyra = Thyra::createVector(v);
  }

  return v_thyra;
}

Teuchos::RCP<const Thyra_Vector> 
createConstThyraVector (const Teuchos::RCP<const Tpetra_Vector> v)
{
  Teuchos::RCP<const Thyra_Vector> v_thyra = Teuchos::null;
  if (!v.is_null()) {
    v_thyra = Thyra::createConstVector(v);
  }

  return v_thyra;
}

Teuchos::RCP<Thyra_MultiVector>
createThyraMultiVector (const Teuchos::RCP<Tpetra_MultiVector> mv)
{
  Teuchos::RCP<Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    mv_thyra = Thyra::createMultiVector(mv);
  }

  return mv_thyra;
}

Teuchos::RCP<const Thyra_MultiVector>
createConstThyraMultiVector (const Teuchos::RCP<const Tpetra_MultiVector> mv)
{
  Teuchos::RCP<const Thyra_MultiVector> mv_thyra = Teuchos::null;
  if (!mv.is_null()) {
    mv_thyra = Thyra::createConstMultiVector(mv);
  }

  return mv_thyra;
}

// ============ Thyra->Tpetra conversion routines ============ //

Teuchos::RCP<Tpetra_Vector>
getTpetraVector (const Teuchos::RCP<Thyra_Vector> v)
{
  Teuchos::RCP<Tpetra_Vector> v_tpetra;
  if (!v.is_null()) {
    v_tpetra = ConverterT::getTpetraVector(v);
  }

  return v_tpetra;
}

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector (const Teuchos::RCP<const Thyra_Vector> v)
{
  Teuchos::RCP<const Tpetra_Vector> v_tpetra;
  if (!v.is_null()) {
    v_tpetra = ConverterT::getConstTpetraVector(v);
  }

  return v_tpetra;
}

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector> mv)
{
  Teuchos::RCP<Tpetra_MultiVector> mv_tpetra;
  if (!mv.is_null()) {
    mv_tpetra = ConverterT::getTpetraMultiVector(mv);
  }

  return mv_tpetra;
}

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv)
{
  Teuchos::RCP<const Tpetra_MultiVector> mv_tpetra;
  if (!mv.is_null()) {
    mv_tpetra = ConverterT::getConstTpetraMultiVector(mv);
  }

  return mv_tpetra;
}

} // namespace Albany
