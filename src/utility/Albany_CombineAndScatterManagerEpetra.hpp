#ifndef ALBANY_COMBINE_AND_SCATTER_MANAGER_EPETRA_HPP
#define ALBANY_COMBINE_AND_SCATTER_MANAGER_EPETRA_HPP

#include "Albany_CombineAndScatterManager.hpp"

#include "Epetra_Import.h"

namespace Albany
{

// This class is a concrete implementation of CombineAndScatterManager,
// for the case where the thyra structures are wrappers of Epetra structures.
// An Epetra_Import object is constructed at construction time, and then reused
// at every combine/scatter call (in either forward or reverse mode).
class CombineAndScatterManagerEpetra : public CombineAndScatterManager
{
public:
  Teuchos::RCP<const Thyra_VectorSpace> getOwnedVectorSpace () const { return owned_vs; }
  Teuchos::RCP<const Thyra_VectorSpace> getOverlappedVectorSpace () const { return overlapped_vs; }

  CombineAndScatterManagerEpetra(const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                                 const Teuchos::RCP<const Thyra_VectorSpace>& overlapped);

  // Combine methods
  void combine (const Thyra_Vector& src,
                      Thyra_Vector& dst,
                const CombineMode CM) const;
  void combine (const Thyra_MultiVector& src,
                      Thyra_MultiVector& dst,
                const CombineMode CM) const;
  void combine (const Thyra_LinearOp& src,
                      Thyra_LinearOp& dst,
                const CombineMode CM) const;

  void combine (const Teuchos::RCP<const Thyra_Vector>& src,
                const Teuchos::RCP<      Thyra_Vector>& dst,
                const CombineMode CM) const;
  void combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
                const Teuchos::RCP<      Thyra_MultiVector>& dst,
                const CombineMode CM) const;
  void combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
                const Teuchos::RCP<      Thyra_LinearOp>& dst,
                const CombineMode CM) const;


  // Scatter methods
  void scatter (const Thyra_Vector& src,
                      Thyra_Vector& dst,
                const CombineMode CM) const;
  void scatter (const Thyra_MultiVector& src,
                      Thyra_MultiVector& dst,
                const CombineMode CM) const;
  void scatter (const Thyra_LinearOp& src,
                      Thyra_LinearOp& dst,
                const CombineMode CM) const;

  void scatter (const Teuchos::RCP<const Thyra_Vector>& src,
                const Teuchos::RCP<      Thyra_Vector>& dst,
                const CombineMode CM) const;
  void scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
                const Teuchos::RCP<      Thyra_MultiVector>& dst,
                const CombineMode CM) const;
  void scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
                const Teuchos::RCP<      Thyra_LinearOp>& dst,
                const CombineMode CM) const;

private:

  Teuchos::RCP<const Thyra_VectorSpace>   owned_vs;
  Teuchos::RCP<const Thyra_VectorSpace>   overlapped_vs;

  Teuchos::RCP<Epetra_Import>   importer;
};

} // namespace Albany

#endif // ALBANY_COMBINE_AND_SCATTER_MANAGER_EPETRA_HPP
