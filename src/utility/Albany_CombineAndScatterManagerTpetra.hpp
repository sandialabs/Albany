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
  void combine (const Thyra_Vector& src,
                      Thyra_Vector& dst,
                const CombineMode CM) const override;
  void combine (const Thyra_MultiVector& src,
                      Thyra_MultiVector& dst,
                const CombineMode CM) const override;
  void combine (const Thyra_LinearOp& src,
                      Thyra_LinearOp& dst,
                const CombineMode CM) const override;

  void combine (const Teuchos::RCP<const Thyra_Vector>& src,
                const Teuchos::RCP<      Thyra_Vector>& dst,
                const CombineMode CM) const override;
  void combine (const Teuchos::RCP<const Thyra_MultiVector>& src,
                const Teuchos::RCP<      Thyra_MultiVector>& dst,
                const CombineMode CM) const override;
  void combine (const Teuchos::RCP<const Thyra_LinearOp>& src,
                const Teuchos::RCP<      Thyra_LinearOp>& dst,
                const CombineMode CM) const override;


  // Scatter methods
  void scatter (const Thyra_Vector& src,
                      Thyra_Vector& dst,
                const CombineMode CM) const override;
  void scatter (const Thyra_MultiVector& src,
                      Thyra_MultiVector& dst,
                const CombineMode CM) const override;
  void scatter (const Thyra_LinearOp& src,
                      Thyra_LinearOp& dst,
                const CombineMode CM) const override;

  void scatter (const Teuchos::RCP<const Thyra_Vector>& src,
                const Teuchos::RCP<      Thyra_Vector>& dst,
                const CombineMode CM) const override;
  void scatter (const Teuchos::RCP<const Thyra_MultiVector>& src,
                const Teuchos::RCP<      Thyra_MultiVector>& dst,
                const CombineMode CM) const override;
  void scatter (const Teuchos::RCP<const Thyra_LinearOp>& src,
                const Teuchos::RCP<      Thyra_LinearOp>& dst,
                const CombineMode CM) const override;

protected:
  void create_ghosted_aura_owners () const override;
  void create_owned_aura_users () const override;

  Teuchos::RCP<Tpetra_Import>   importer;
};

} // namespace Albany

#endif // ALBANY_COMBINE_AND_SCATTER_MANAGER_TPETRA_HPP
