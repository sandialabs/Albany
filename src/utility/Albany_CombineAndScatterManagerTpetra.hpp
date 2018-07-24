#ifndef ALBANY_COMBINE_AND_SCATTER_MANAGER_TPETRA_HPP
#define ALBANY_COMBINE_AND_SCATTER_MANAGER_TPETRA_HPP

#include "Albany_CombineAndScatterManager.hpp"

#include "Albany_TpetraTypes.hpp"

namespace Albany
{

// This class is a concrete implementation of CombineAndScatterManager,
// for the case where the thyra structures are wrappers of Tpetra structures.
// An Tpetra_Import object is constructed at construction time, and then reused
// at every combine/scatter call (in either forward or reverse mode).
class CombineAndScatterManagerTpetra : public CombineAndScatterManager
{
public:
  CombineAndScatterManagerTpetra(const Teuchos::RCP<const Thyra_VectorSpace>& owned,
                                 const Teuchos::RCP<const Thyra_VectorSpace>& overlapped);

  // Combine methods
  void combine (const Teuchos::RCP<const Thyra_Vector>& src,
                const Teuchos::RCP<Thyra_Vector>& dst,
                const CombineMode CM) const;
  void combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
                const Teuchos::RCP<Thyra_MultiVector>& dst,
                const CombineMode CM) const;
  void combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
                const Teuchos::RCP<Thyra_LinearOp>& dst,
                const CombineMode CM) const;

  // Scatter methods
  void scatter (const Teuchos::RCP<const Thyra_Vector>& src,
                const Teuchos::RCP<Thyra_Vector>& dst,
                const CombineMode CM) const;
  void scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
                const Teuchos::RCP<Thyra_MultiVector>& dst,
                const CombineMode CM) const;
  void scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
                const Teuchos::RCP<Thyra_LinearOp>& dst,
                const CombineMode CM) const;

private:

  Teuchos::RCP<Tpetra_Import>   importer;
};

} // namespace Albany

#endif // ALBANY_COMBINE_AND_SCATTER_MANAGER_TPETRA_HPP
