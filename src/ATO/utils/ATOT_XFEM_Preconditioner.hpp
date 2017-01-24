#ifndef ATOT_XFEM_PRECONDITIONER_HPP
#define ATOT_XFEM_PRECONDITIONER_HPP

#include <Tpetra_Operator.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Albany_AbstractDiscretization.hpp>
#include <Albany_StateManager.hpp>

#include <string>


namespace ATOT {
namespace XFEM {
  using Teuchos::RCP;

  /** \brief
   */
  class Preconditioner : public Tpetra_Operator
  {
  public:
    /** */
    Preconditioner(const RCP<const Teuchos::ParameterList> params);

    /** */
    virtual ~Preconditioner() {;}

    /** */
    int SetUseTranspose(bool useTranspose) {
      useTranspose_ = useTranspose;
      return 0;
    }

    /** */
    int BuildPreconditioner(
           const RCP<Tpetra_CrsMatrix>& jacT, 
           const RCP<Albany::AbstractDiscretization>& disc,
           const Albany::StateManager& stateMgr);

    /** */
    void apply(
      Tpetra_MultiVector const & X,
      Tpetra_MultiVector & Y,
      Teuchos::ETransp mode = Teuchos::NO_TRANS,
      ST alpha = Teuchos::ScalarTraits<ST>::one(),
      ST beta = Teuchos::ScalarTraits<ST>::zero()) const;

    /** */
    bool hasTransposeApply() const {return useTranspose_;}

    /** */
    Teuchos::RCP<const Tpetra_Map> getDomainMap() const {return domainMap_;} 

    /** */
    Teuchos::RCP<const Tpetra_Map> getRangeMap() const {return rangeMap_;}

  protected:
    /** */
    Preconditioner();

    /** */
    bool useTranspose_;

    /** */
    std::string label_;

    /** */
    RCP<const Tpetra_Map> domainMap_;

    /** */
    RCP<const Tpetra_Map> rangeMap_;

    /** */
    RCP<Tpetra_CrsMatrix> operator_;
    RCP<Tpetra_CrsMatrix> invOperator_;

  };
} // end namespace XFEM
} // end namespace ATOT

#endif
