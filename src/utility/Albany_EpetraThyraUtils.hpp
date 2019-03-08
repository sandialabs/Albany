#ifndef ALBANY_EPETRA_THYRA_UTILS_HPP
#define ALBANY_EPETRA_THYRA_UTILS_HPP

#include "Albany_ThyraTypes.hpp"

#include "Epetra_Comm.h"
#include "Epetra_BlockMap.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"

// The type of an Epetra global id. Epetra uses long long for their 64 bits integers
using Epetra_GO = long long;


namespace Albany
{

// The wrappers in thyra throw if the input Thyra/Epetra pointer is null
// These routines are here to handle that case, and simply return a
// Teuchos::null if the input RCP is null. They are just a convenience
// routine that performs the check before calling the Thyra converter.

// ============ Epetra->Thyra conversion routines ============ //

Teuchos::RCP<const Thyra_SpmdVectorSpace>
createThyraVectorSpace (const Teuchos::RCP<const Epetra_BlockMap> map);

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

// ============ Thyra->Epetra conversion routines ============ //

Teuchos::RCP<const Epetra_BlockMap>
getEpetraBlockMap (const Teuchos::RCP<const Thyra_VectorSpace> vs,
                   const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_Map>
getEpetraMap (const Teuchos::RCP<const Thyra_VectorSpace> vs,
              const bool throw_on_failure = true);

Teuchos::RCP<Epetra_Vector>
getEpetraVector (const Teuchos::RCP<Thyra_Vector> v,
                 const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_Vector>
getConstEpetraVector (const Teuchos::RCP<const Thyra_Vector> v,
                      const bool throw_on_failure = true);

Teuchos::RCP<Epetra_MultiVector>
getEpetraMultiVector (const Teuchos::RCP<Thyra_MultiVector> mv,
                      const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_MultiVector>
getConstEpetraMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv,
                           const bool throw_on_failure = true);

Teuchos::RCP<Epetra_Operator>
getEpetraOperator (const Teuchos::RCP<Thyra_LinearOp> lop,
                   const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_Operator>
getConstEpetraOperator (const Teuchos::RCP<const Thyra_LinearOp> lop,
                        const bool throw_on_failure = true);

Teuchos::RCP<Epetra_CrsMatrix>
getEpetraMatrix (const Teuchos::RCP<Thyra_LinearOp> lop,
                 const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_CrsMatrix>
getConstEpetraMatrix (const Teuchos::RCP<const Thyra_LinearOp> lop,
                      const bool throw_on_failure = true);

// --- Conversion from references rather than RCPs --- //
// Note: return pointers cause we want to allow failure,
//       in which case we will return nullptr
// Note: in this case, we have no way to retrieve the original
//       Epetra objects, since there is no Thyra::EpetraVector
//       (and the likes) class, (except for the LinearOp case).
//       The thyra adapters, simply create an SPMD object RCP,
//       then attach the original Thyra RCP as extra data.
//       Here, we do not have an RCP, so no extra data.
//       The only thing left to do is to create an Epetra_(Multi)Vector
//       that views the local data of the Thyra_(Multi)Vector.
//       To do this, we *need* the map, so we *need* an Epetra_Map
//       passed in as input. Notice that we could always do this,
//       regardless of whether the Thyra_(Multi)Vector was indeed
//       created from an Epetra_(Multi)Vector, since we can always
//       view its local data. For this reason, we first perform
//       a cast to make sure the concrete type of the object is
//       Thyra::DefaultSpmd(Multi)Vector<ST> (and avoid the case
//       where the input (Multi)Vector was not created from an
//       Epetra (Multi)Vector).
//       Finally, notice that, since we are not simply casting
//       the input, but are actually *creating* an object (albeit
//       simply *viewing* the input's values), we cannot return
//       a pointer or reference for the (Multi)Vector conversions.
//       We therefore return an RCP.

Teuchos::RCP<Epetra_Vector>
getEpetraVector (Thyra_Vector& v,
                 const Epetra_BlockMap& emap,
                 const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_Vector>
getConstEpetraVector (const Thyra_Vector& v,
                      const Epetra_BlockMap& emap,
                      const bool throw_on_failure = true);

Teuchos::RCP<Epetra_MultiVector>
getEpetraMultiVector (Thyra_MultiVector& mv,
                      const Epetra_BlockMap& emap,
                      const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_MultiVector>
getConstEpetraMultiVector (const Thyra_MultiVector& mv,
                           const Epetra_BlockMap& emap,
                           const bool throw_on_failure = true);

Teuchos::RCP<Epetra_Operator>
getEpetraOperator (Thyra_LinearOp& lop,
                   const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_Operator>
getConstEpetraOperator (const Thyra_LinearOp& lop,
                        const bool throw_on_failure = true);

Teuchos::RCP<Epetra_CrsMatrix>
getEpetraMatrix (Thyra_LinearOp& lop,
                 const bool throw_on_failure = true);

Teuchos::RCP<const Epetra_CrsMatrix>
getConstEpetraMatrix (const Thyra_LinearOp& lop,
                      const bool throw_on_failure = true);

} // namespace Albany

#endif // ALBANY_EPETRA_THYRA_UTILS_HPP
