#ifndef ALBANY_EPETRA_THYRA_UTILS_HPP
#define ALBANY_EPETRA_THYRA_UTILS_HPP

#include "Albany_ThyraTypes.hpp"

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"

namespace Albany
{

// The wrappers in thyra throw if the input Thyra/Tpetra pointer is null
// These routines are here to handle that case, and simply return a
// Teuchos::null if the input RCP is null. They are just a convenience
// routine that performs the check before calling the Thyra converter.

// ============ Tpetra->Thyra conversion routines ============ //
Teuchos::RCP<const Thyra_VectorSpace>
createThyraVectorSpace (const Teuchos::RCP<const Epetra_Map> map);

Teuchos::RCP<Thyra_Vector>
createThyraVector (const Teuchos::RCP<Epetra_Vector> v);

Teuchos::RCP<const Thyra_Vector>
createConstThyraVector (const Teuchos::RCP<const Epetra_Vector> v);

Teuchos::RCP<Thyra_MultiVector>
createThyraMultiVector (const Teuchos::RCP<Epetra_MultiVector> mv);

Teuchos::RCP<const Thyra_MultiVector>
createConstThyraMultiVector (const Teuchos::RCP<const Epetra_MultiVector> mv);

Teuchos::RCP<Thyra_LinearOp>
createThyraLinearOp (const Teuchos::RCP<Epetra_Operator> op);

Teuchos::RCP<const Thyra_LinearOp>
createConstThyraLinearOp (const Teuchos::RCP<const Epetra_Operator> op);

// ============ Thyra->Tpetra conversion routines ============ //
Teuchos::RCP<const Epetra_Map>
getEpetraMap (const Teuchos::RCP<const Thyra_VectorSpace> vs,
              const bool throw_on_failure = true);

Teuchos::RCP<Epetra_Vector>
getEpetraVector (const Teuchos::RCP<Thyra_Vector> v,
                 const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_Vector>
getConstTpetraVector (const Teuchos::RCP<const Thyra_Vector> v,
                      const bool throw_on_failure = true);

Teuchos::RCP<Epetra_MultiVector>
getEpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector> mv,
                      const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_MultiVector>
getConstTpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv,
                           const bool throw_on_failure = true);

Teuchos::RCP<Epetra_Operator>
getEpetraOperator (const Teuchos::RCP<Thyra_LinearOp> lop,
                   const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_Operator>
getConstTpetraOperator (const Teuchos::RCP<const Thyra_LinearOp> lop,
                        const bool throw_on_failure = true);

Teuchos::RCP<Epetra_CrsMatrix>
getEpetraMatrix (const Teuchos::RCP<Thyra_LinearOp> lop,
                 const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_CrsMatrix>
getConstTpetraMatrix (const Teuchos::RCP<const Thyra_LinearOp> lop,
                      const bool throw_on_failure = true);

} // namespace Albany

#endif // ALBANY_EPETRA_THYRA_UTILS_HPP
