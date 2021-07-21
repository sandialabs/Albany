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
Teuchos::RCP<const Thyra_SpmdVectorSpace>
createThyraVectorSpace (const Teuchos::RCP<const Tpetra_Map>& map);

Teuchos::RCP<Thyra_Vector>
createThyraVector (const Teuchos::RCP<Tpetra_Vector>& v);

Teuchos::RCP<const Thyra_Vector>
createConstThyraVector (const Teuchos::RCP<const Tpetra_Vector>& v);

Teuchos::RCP<Thyra_MultiVector>
createThyraMultiVector (const Teuchos::RCP<Tpetra_MultiVector>& mv);

Teuchos::RCP<const Thyra_MultiVector>
createConstThyraMultiVector (const Teuchos::RCP<const Tpetra_MultiVector>& mv);

Teuchos::RCP<Thyra_LinearOp>
createThyraLinearOp (const Teuchos::RCP<Tpetra_Operator>& op);

Teuchos::RCP<Thyra_BlockedLinearOp>
createThyraBlockedLinearOp (const Teuchos::RCP<Tpetra_Operator>& op);

Teuchos::RCP<const Thyra_LinearOp>
createConstThyraLinearOp (const Teuchos::RCP<const Tpetra_Operator>& op);

// ============ Thyra->Tpetra conversion routines ============ //
Teuchos::RCP<const Tpetra_Map>
getTpetraMap (const Teuchos::RCP<const Thyra_VectorSpace>& vs,
              const bool throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_Vector>
getTpetraVector (const Teuchos::RCP<Thyra_Vector>& v,
                 const bool throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector (const Teuchos::RCP<const Thyra_Vector>& v,
                      const bool throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector>& mv,
                      const bool throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector>& mv,
                           const bool throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_Operator>
getTpetraOperator (const Teuchos::RCP<Thyra_LinearOp>& lop,
                   const bool throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_Operator>
getConstTpetraOperator (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                        const bool throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_CrsMatrix>
getTpetraMatrix (const Teuchos::RCP<Thyra_LinearOp>& lop,
                 const bool throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_CrsMatrix>
getConstTpetraMatrix (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                      const bool throw_if_not_tpetra = true);

// Tpetra_FECrsMatrix has changed the way we modify its entries.
// See https://github.com/trilinos/trilinos/pull/9000
Teuchos::RCP<Tpetra_FECrsMatrix>
getTpetraFECrsMatrix (const Teuchos::RCP<Thyra_LinearOp>& lop,
                      const bool throw_if_not_tpetra);

// --- Conversion from references rather than RCPs --- //

Teuchos::RCP<Tpetra_Vector>
getTpetraVector (Thyra_Vector& v,
                 const bool throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_Vector>
getConstTpetraVector (const Thyra_Vector& v,
                      const bool throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_MultiVector>
getTpetraMultiVector (Thyra_MultiVector& mv,
                      const bool throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_MultiVector>
getConstTpetraMultiVector (const Thyra_MultiVector& mv,
                           const bool throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_Operator>
getTpetraOperator (Thyra_LinearOp& lop,
                   const bool throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_Operator>
getConstTpetraOperator (const Thyra_LinearOp& lop,
                        const bool throw_if_not_tpetra = true);

Teuchos::RCP<Tpetra_CrsMatrix>
getTpetraMatrix (Thyra_LinearOp& lop,
                 const bool throw_if_not_tpetra = true);

Teuchos::RCP<const Tpetra_CrsMatrix>
getConstTpetraMatrix (const Thyra_LinearOp& lop,
                      const bool throw_if_not_tpetra = true);

} // namespace Albany

#endif // ALBANY_TPETRA_THYRA_UTILS_HPP
