#ifndef ATO_XFEM_PRECONDITIONER_HPP
#define ATO_XFEM_PRECONDITIONER_HPP

#include <Epetra_Map.h>
#include <Epetra_Comm.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Operator.h>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Albany_AbstractDiscretization.hpp>
#include <Albany_StateManager.hpp>

#include <string>


namespace ATO {
namespace XFEM {
  using Teuchos::RCP;

  /** \brief
   */
  class Preconditioner : public Epetra_Operator
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
           const RCP<Epetra_CrsMatrix>& jac, 
           const RCP<Albany::AbstractDiscretization>& disc,
           const Albany::StateManager& stateMgr);

    /** */
    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const ;

    /** */
    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const ;

    /** */
    double NormInf() const ;

    /** */
    const char* Label() const {return label_.c_str();}

    /** */
    bool UseTranspose() const {return useTranspose_;}

    /** */
    bool HasNormInf() const {return false;}

    /** */
    const Epetra_Comm & Comm() const {return *comm_;}

    /** */
    const Epetra_Map& OperatorDomainMap() const {return *domainMap_;}

    /** */
    const Epetra_Map& OperatorRangeMap() const {return *rangeMap_;}

  protected:
    /** */
    Preconditioner();

    /** */
    bool useTranspose_;

    /** */
    RCP<const Epetra_Comm> comm_;

    /** */
    std::string label_;

    /** */
    RCP<const Epetra_Map> domainMap_;

    /** */
    RCP<const Epetra_Map> rangeMap_;

    /** */
    RCP<Epetra_CrsMatrix> operator_;
    RCP<Epetra_CrsMatrix> invOperator_;

  };
} // end namespace XFEM
} // end namespace ATO

#endif
