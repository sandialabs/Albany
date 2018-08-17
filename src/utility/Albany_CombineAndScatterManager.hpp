#ifndef ALBANY_COMBINE_AND_SCATTER_MANAGER_HPP
#define ALBANY_COMBINE_AND_SCATTER_MANAGER_HPP

#include "Albany_ThyraTypes.hpp"

namespace Albany
{

// An Albany-owned combine mode enum, so we don't stick to either Tpetra or Epetra
// (note:: it would be nice if Teuchos provided such enumeration)
enum class CombineMode {
  ADD,    // Add remote contributions to local ones
  INSERT, // Replace local contributions with remote ones (beware of race conditions!)
  ZERO    // Ignore remote contributions
};

// This class is intended to hide the implementation details regarding
// how the underlying linear algebra library deals with combining/scattering
// distributed objects. The interface accepts Thyra objects (vectors,
// multivectors, linear operators), and in the hidden implementation,
// these are casted to the concrete linear algebra structures, and
// the corresponding combine/scatter method is used
class CombineAndScatterManager
{
public:
  virtual Teuchos::RCP<const Thyra_VectorSpace> getOwnedVectorSpace () const = 0;
  virtual Teuchos::RCP<const Thyra_VectorSpace> getOverlappedVectorSpace () const = 0;

  // Combine methods
  virtual void combine (const Teuchos::RCP<const Thyra_Vector>& src,
                        const Teuchos::RCP<Thyra_Vector>& dst,
                        const CombineMode CM) const = 0;
  virtual void combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
                        const Teuchos::RCP<Thyra_MultiVector>& dst,
                        const CombineMode CM) const = 0;
  virtual void combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
                        const Teuchos::RCP<Thyra_LinearOp>& dst,
                        const CombineMode CM) const = 0;

  // Scatter methods
  virtual void scatter (const Teuchos::RCP<const Thyra_Vector>& src,
                        const Teuchos::RCP<Thyra_Vector>& dst,
                        const CombineMode CM) const = 0;
  virtual void scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
                        const Teuchos::RCP<Thyra_MultiVector>& dst,
                        const CombineMode CM) const = 0;
  virtual void scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
                        const Teuchos::RCP<Thyra_LinearOp>& dst,
                        const CombineMode CM) const = 0;
};

// Utility function that returns a concrete manager, depending on the return value
// of Albany::build_type().
Teuchos::RCP<CombineAndScatterManager>
createCombineAndScatterManager (const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                                const Teuchos::RCP<const Thyra_VectorSpace>& overlapped);

} // namespace Albany

#endif // ALBANY_COMBINE_AND_SCATTER_MANAGER_HPP
