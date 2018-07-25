#ifndef ALBANY_THYRA_UTILS_HPP
#define ALBANY_THYRA_UTILS_HPP

// Get basic Thyra types
#include "Albany_ThyraTypes.hpp"

// Get Product Thyra types
#include "Thyra_ProductVectorSpaceBase.hpp"
#include "Thyra_ProductMultiVectorBase.hpp"
#include "Thyra_ProductVectorBase.hpp"

// Product types
typedef Thyra::ProductVectorSpaceBase<ST>   Thyra_ProductVectorSpace;
typedef Thyra::ProductMultiVectorBase<ST>   Thyra_ProductMultiVector;
typedef Thyra::ProductVectorBase<ST>        Thyra_ProductVector;

namespace Albany
{

// These routines help to manipulate the a Thyra_LinearOp. They are needed
// so we can abstract from the concrete linear algebra package, and rely
// only on the Thyra interfaces.

bool isFillActive (const Teuchos::RCP<const Thyra_LinearOp>& lop);
void resumeFill (const Teuchos::RCP<Thyra_LinearOp>& lop);
void fillComplete (const Teuchos::RCP<Thyra_LinearOp>& lop);
void assign (const Teuchos::RCP<Thyra_LinearOp>& lop, const ST value);
void getDiagonalCopy (const Teuchos::RCP<const Thyra_LinearOp>& lop,
                      Teuchos::RCP<Thyra_Vector>& diag);

// TODO: remove this when Trilinos issue #3180 is resolved
// This routine would not be needed if Thyra::SpmdVectorBase's method
// getNonconstLocalData did the right thing. However, the concrete
// implementation Thyra::TpetraVector calls get1dViewNonConst from
// Tpetra::MultiVector, which does not mark the host view as modified.
// Therefore, the modifications to the vector are never synced to device.
// Until this behavior is fixed, we use our own implementation.
Teuchos::ArrayRCP<ST> getNonconstLocalData (const Teuchos::RCP<Thyra_Vector>& v);

// We make this one available too, so that the const/nonconst versions come from the same place
Teuchos::ArrayRCP<const ST> getLocalData (const Teuchos::RCP<const Thyra_Vector>& v);

// This is just a utility routine, that mildly extend the update method of Thyra_Vector,
// but does not have the complex signature of the linear_combination method of Thyra_Vector.
// In fact, the update method only allows to do y = y + alpha*x, while often one wants
// to do y = beta*y + alpha*x. The linear_combination method offers that capability,
// but the signature is more cumbersome. Thyra offers also a free function with a lighter
// signature for a linear_combination of 2 vectors, but its name is V_StVpStV, which
// can be a bit arcane. Here we simply wrap that function in one with a nicer name.
// Performs x = y_coeff*y + x_coeff*x;
void scale_and_update (const Teuchos::RCP<Thyra_Vector> y, const ST y_coeff,
                       const Teuchos::RCP<const Thyra_Vector> x, const ST x_coeff);

// ========= Matrix Market utilities ========== //

void
writeMatrixMarket(
    const Teuchos::RCP<const Thyra_Vector>& v, const std::string& prefix,
    int const counter = -1);

void
writeMatrixMarket(
    const Teuchos::Array<Teuchos::RCP<const Thyra_Vector>>& vs,
    const std::string& prefix, int const counter = -1);

void
writeMatrixMarket(
    const Teuchos::RCP<const Thyra_LinearOp>& A, const std::string& prefix,
    int const counter = -1);

void
writeMatrixMarket(
    const Teuchos::Array<Teuchos::RCP<const Thyra_LinearOp>>& As,
    const std::string& prefix, int const counter = -1);

// ========= Thyra_ProductXYZ utilities ========== //

// These routines help to manipulate Thyra pointers, casting them to
// proper derived classes, and checking that the result is nonnull.

Teuchos::RCP<const Thyra_ProductVectorSpace>
getProductVectorSpace (const Teuchos::RCP<const Thyra_VectorSpace> vs,
                       const bool throw_on_failure = true);

Teuchos::RCP<Thyra_ProductVector>
getProductVector (const Teuchos::RCP<Thyra_Vector> v,
                  const bool throw_on_failure = true);

Teuchos::RCP<const Thyra_ProductVector>
getConstProductVector (const Teuchos::RCP<const Thyra_Vector> v,
                       const bool throw_on_failure = true);

Teuchos::RCP<Thyra_ProductMultiVector>
getProductMultiVector (const Teuchos::RCP<Thyra_MultiVector> mv,
                       const bool throw_on_failure = true);

Teuchos::RCP<const Thyra_ProductMultiVector>
getConstProductMultiVector (const Teuchos::RCP<const Thyra_MultiVector> mv,
                            const bool throw_on_failure = true);

} // namespace Albany

#endif // ALBANY_THYRA_UTILS_HPP
