#ifndef ALBANY_HESSIAN_HPP
#define ALBANY_HESSIAN_HPP

#include "MatrixMarket_Tpetra.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "Albany_TpetraThyraUtils.hpp"
#include "Tpetra_Core.hpp"
#include "Albany_Application.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Thyra_Ifpack2PreconditionerFactory.hpp"
#include "Teuchos_AbstractFactoryStd.hpp"
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"

namespace Albany
{
    /**
     * \brief createDenseHessianLinearOp function
     *
     * This function computes the Thyra::LinearOp associated to
     * the Hessian w.r.t a parameter vector.
     *
     * \param p_vs [in] Thyra::VectorSpace which specifies the entries of the current parameter vector.
     */
    Teuchos::RCP<Thyra_LinearOp> createDenseHessianLinearOp(
        Teuchos::RCP<const Thyra_VectorSpace> p_vs);

    /**
     * \brief createSparseHessianLinearOp function
     *
     * This function computes the Thyra::LinearOp associated to
     * the Hessian w.r.t a distributed parameter.
     *
     * \param p_owned_vs [in] Thyra::VectorSpace which specifies the owned entries of the current distributed parameter.
     *
     * \param p_overlapped_vs [in] Thyra::VectorSpace which specifies the overlapped entries of the current distributed parameter.
     *
     * \param wsElDofs [in] Vector of IDArray associated to the mesh used.
     */
    Teuchos::RCP<Thyra_LinearOp> createSparseHessianLinearOp(
        Teuchos::RCP<const Thyra_VectorSpace> p_owned_vs,
        Teuchos::RCP<const Thyra_VectorSpace> p_overlapped_vs,
        const std::vector<IDArray> wsElDofs);

    /**
     * \brief getHessianBlockIDs function
     *
     * This function gets the block IDs of a block of the Hessian matrix from a blockName.
     * For example, this function associates the following IDs to the following names:
     *  - i1=0, i2=0, and blockName="(x,x)",
     *  - i1=0, i2=1, and blockName="(x,p0)",
     *  - i1=0, i2=2, and blockName="(x,p1)",
     *  - i1=1, i2=0, and blockName="(p0,x)",
     *  - i1=3, i2=5, and blockName="(p2,p4)",
     *  - (...)
     * 
     * \param i1 [out] First block index.
     *
     * \param i2 [out] Second block index.
     *
     * \param blockName [in] Name of the block for which the IDs have to be computed.
     */
    void getHessianBlockIDs(
        int & i1,
        int & i2,
        std::string blockName
    );

    /**
     * \brief getParameterVectorID function 
     *
     * This function tests if the name parameterName is associated to a parameter vector
     * (in this case, a scalar parameter which is not included in a parameter vector in the .yaml file
     * is considered as a parameter vector of size 1) or to a distributed parameter.
     *
     * If the parameterName is associated to a parameter vector, the function computes the index of
     * associated parameter vector.
     *
     * \param i [out] Index associated to the parameter vector (if the current parameter is a parameter vector).
     *
     * \param is_distributed [out] Bool which specifies if the current parameter is a parameter vector.
     *
     * \param parameterName [in] Name of the current parameter.
     */
    void getParameterVectorID(
        int & i,
        bool & is_distributed,
        std::string parameterName
    );



    //! Thyra_LinearOp implementing the action of the Hessian d^2 g / dpp
    /*!
     * This class implements the Thyra::LinearOpBase interface for
     * op(df/dp)*v where op() is the identity or tranpose, f is the Albany
     * residual vector, p is a distributed parameter vector, and v is a given
     * vector.
     */
    class AtA_LinearOp : public Thyra_LOWS {
    public:

      // Constructor
      AtA_LinearOp(
        const Teuchos::RCP<Application>& app,
        const Teuchos::ParameterList &responseParams) :
        app_(app)
        {
          coeff_ = responseParams.get<double>("Scaling Coefficient");
          field_name_ = responseParams.get<std::string>("Field Name");
          file_name_A_ = responseParams.get<std::string>("Linear Operator File Name");
          file_name_D_ = responseParams.get<std::string>("Diagonal Scaling File Name");
          param_name_ = "";
        }

      //! Destructor
      virtual ~AtA_LinearOp() {}


      //! Overrides Thyra::LinearOpBase purely virtual method
      Teuchos::RCP<const Thyra_VectorSpace> domain() const {
        return app_->getDistributedParameterLibrary()->get(field_name_)->vector_space();
      }

      //! Overrides Thyra::LinearOpBase purely virtual method
      Teuchos::RCP<const Thyra_VectorSpace> range() const {
        return app_->getDistributedParameterLibrary()->get(field_name_)->vector_space();
      }

      void setup(const std::string& param_name) {
        param_name_ = param_name;
        if(A_.is_null() && (field_name_ == param_name_))
          AtA_LinearOp::loadLinearOperator();
        Teuchos::ParameterList pl = app_->getAppPL()->sublist("Piro").sublist("Analysis").sublist("ROL").sublist("Matrix Based Dot Product").sublist("Matrix Types").sublist("Hessian Of Response").sublist("Block Diagonal Solver").sublist(util::strint("Block",0));
        std::string solverType = pl.get<std::string>("Linear Solver Type");
        pl.remove("Use Custom Solve",false); //parameter not valid for solvers

        Stratimikos::DefaultLinearSolverBuilder strat;
        typedef Thyra::PreconditionerFactoryBase<ST> Base;
        typedef Thyra::Ifpack2PreconditionerFactory<Tpetra_CrsMatrix> Impl;
        strat.setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
        strat.setParameterList(Teuchos::rcpFromRef(pl));
        Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<double> > lows_factory = strat.createLinearSolveStrategy(solverType);
        Teuchos::RCP< const ::Thyra::DefaultLinearOpSource<double> > losb = Teuchos::rcp(new ::Thyra::DefaultLinearOpSource<double>(A_));
        Teuchos::RCP< ::Thyra::PreconditionerBase<double> > prec;

        Teuchos::RCP< ::Thyra::PreconditionerFactoryBase<double> > prec_factory =  lows_factory->getPreconditionerFactory();
        if (Teuchos::nonnull(prec_factory)) {
          prec = prec_factory->createPrec();
        }

        A_solve_ = lows_factory->createOp();
        Atrans_solve_ = lows_factory->createOp();

        if (Teuchos::nonnull(prec_factory))
          prec_factory->initializePrec(losb, prec.get());

        if(Teuchos::nonnull(prec)) {
          Thyra::initializePreconditionedOp<double>(*lows_factory,
              Thyra::transpose<double>(A_),
              Thyra::unspecifiedPrec<double>(::Thyra::transpose<double>(prec->getUnspecifiedPrecOp())),
              A_solve_.ptr());

          Thyra::initializePreconditionedOp<double>(*lows_factory,
              A_,
              prec,
              Atrans_solve_.ptr());

        }
        else {
          Thyra::initializeOp<double>(*lows_factory, A_, A_solve_.ptr());
          Thyra::initializeOp<double>(*lows_factory, Thyra::transpose<double>(A_), Atrans_solve_.ptr());
        }
      }

    private:
      void
      loadLinearOperator() {
        Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();
        Teuchos::RCP<const Tpetra_Map> rowMap = Albany::getTpetraMap(field->space());

        Teuchos::RCP<const Tpetra_Map> colMap;
        Teuchos::RCP<const Tpetra_Map> domainMap = rowMap;
        Teuchos::RCP<const Tpetra_Map> rangeMap = rowMap;
        typedef Tpetra::MatrixMarket::Reader<Tpetra_CrsMatrix> reader_type;

        bool mapIsContiguous =
            (static_cast<Tpetra_GO>(rowMap->getMaxAllGlobalIndex()+1-rowMap->getMinAllGlobalIndex()) ==
             static_cast<Tpetra_GO>(rowMap->getGlobalNumElements()));

        TEUCHOS_TEST_FOR_EXCEPTION (!mapIsContiguous, std::runtime_error,
                                    "Error! Row Map needs to be contigous for the Matrix reader to work.\n");

        auto tpetra_mat =
            reader_type::readSparseFile (file_name_A_, rowMap, colMap, domainMap, rangeMap);

        auto tpetra_diag_mat =
            reader_type::readSparseFile (file_name_D_, rowMap, colMap, domainMap, rangeMap);
        Teuchos::RCP<Tpetra_Vector> tpetra_diag_vec = Teuchos::rcp(new Tpetra_Vector(rowMap));
        tpetra_diag_mat->getLocalDiagCopy (*tpetra_diag_vec);

        A_ = Albany::createThyraLinearOp(tpetra_mat);
        D_ = Albany::createThyraVector(tpetra_diag_vec);
        vec1_ = Thyra::createMember(A_->range());
        vec2_ = Thyra::createMember(A_->range());
      }

      //@}

    protected:
      //! Overrides Thyra::LinearOpBase purely virtual method
      bool opSupportedImpl(Thyra::EOpTransp /*M_trans*/) const {
        // The underlying scalar type is not complex, and we support transpose, so we support everything.
        return true;
      }

      //! Overrides Thyra::LinearOpBase purely virtual method
      void applyImpl (const Thyra::EOpTransp /*M_trans*/, //matrix is symmetric
                      const Thyra_MultiVector& X,
                      const Teuchos::Ptr<Thyra_MultiVector>& Y,
                      const ST /* alpha */,
                      const ST /* beta */) const {
        if(field_name_ != param_name_) {
          Y->assign(0.0);
          return;
        }

        Teuchos::RCP<const Thyra_Vector> field = app_->getDistributedParameterLibrary()->get(field_name_)->vector();

        //A p
        A_->apply(Thyra::EOpTransp::NOTRANS, X, vec1_.ptr(), 1.0, 0.0);

        // 2 coeff inv(D) A p
        vec2_->assign(0.0);
        Thyra::ele_wise_divide( 2.0*coeff_, *vec1_, *D_, vec2_.ptr() );

        // 2 coeff A' inv(D) A p
        A_->apply(Thyra::EOpTransp::TRANS, *vec2_, Y, 1.0, 0.0);
      }

      Thyra::SolveStatus<double> solveImpl(
        const Thyra::EOpTransp transp,
        const Thyra_MultiVector &B,
        const Teuchos::Ptr<Thyra_MultiVector> &X,
        const Teuchos::Ptr<const Thyra::SolveCriteria<ST> > solveCriteria
        ) const {
        Thyra::SolveStatus<double> solveStatus;
        if(field_name_ != param_name_) {
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Trying to invert a singular operator.\n");
        } else {
          Thyra::SolveStatus<double> solveStatus1, solveStatus2;
          solveStatus1 = Atrans_solve_->solve(Thyra::EOpTransp::NOTRANS, B, vec1_.ptr(), solveCriteria);
          vec2_->assign(0.0);
          Thyra::ele_wise_prod( 0.5/coeff_, *vec1_, *D_, vec2_.ptr() );
          solveStatus2 = A_solve_->solve(Thyra::EOpTransp::NOTRANS, *vec2_, X, solveCriteria);

          if((solveStatus1.solveStatus == Thyra::SOLVE_STATUS_CONVERGED) && (solveStatus2.solveStatus == Thyra::SOLVE_STATUS_CONVERGED))
            solveStatus.solveStatus =  Thyra::SOLVE_STATUS_CONVERGED;
          else if ((solveStatus1.solveStatus == Thyra::SOLVE_STATUS_UNCONVERGED) || (solveStatus2.solveStatus == Thyra::SOLVE_STATUS_UNCONVERGED))
            solveStatus.solveStatus =  Thyra::SOLVE_STATUS_UNCONVERGED;
        }

        return solveStatus;
      }

      //! Albany applications
      Teuchos::RCP<Application> app_;

      //! Name of distributed parameter we are differentiating w.r.t.
      std::string param_name_;


      Teuchos::RCP<Thyra_LinearOp> A_, Ainv_;
      Teuchos::RCP<Thyra_LOWS> A_solve_, Atrans_solve_;
      std::string field_name_, file_name_A_, file_name_D_;

      Teuchos::RCP<Thyra_Vector> D_,g_,vec1_,vec2_;
      double coeff_;
      Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> > lowsFactory_;

      //@}

    }; // class DistributedParameterDerivativeOp


} // namespace Albany

#endif // ALBANY_HESSIAN_HPP
