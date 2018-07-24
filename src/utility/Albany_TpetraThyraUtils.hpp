#ifndef ALBANY_TPETRA_THYRA_UTILS_HPP
#define ALBANY_TPETRA_THYRA_UTILS_HPP

#include "Albany_TpetraTypes.hpp"
#include "Albany_ThyraTypes.hpp"

namespace Albany
{

// The wrappers in thyra throw if the input Thyra/Tpetra pointer is null
// These routines are here to handle that case, and simply return a
// Teuchos::null if the input RCP is null. They are just a convenience
// routine that performs the check before calling the Thyra converter.

// ============ Tpetra->Thyra conversion routines ============ //
Teuchos::RCP<Thyra_Vector>
createThyraVector (const Teuchos::RCP<Tpetra_Vector> v);

Teuchos::RCP<const Thyra_Vector>
createConstThyraVector (const Teuchos::RCP<const Tpetra_Vector> v);

Teuchos::RCP<Thyra_MultiVector>
createThyraMultiVector (const Teuchos::RCP<Tpetra_MultiVector> mv);

Teuchos::RCP<const Thyra_MultiVector>
createConstThyraMultiVector (const Teuchos::RCP<const Tpetra_MultiVector> mv);

// ============ Thyra->Tpetra conversion routines ============ //
Teuchos::RCP<Tpetra_Vector>
getTpetraVector (const Teuchos::RCP<Thyra_Vector> v);

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector (const Teuchos::RCP<const Thyra_Vector> v);

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector> mv);

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv);

} // namespace Albany

#endif // ALBANY_TPETRA_THYRA_UTILS_HPP
