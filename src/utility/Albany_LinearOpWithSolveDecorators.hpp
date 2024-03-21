#ifndef ALBANY_LINEAROPWITHSOLVEDECORATORS_HPP
#define ALBANY_LINEAROPWITHSOLVEDECORATORS_HPP


#include "Albany_ThyraTypes.hpp"
#include "Thyra_Ifpack2PreconditionerFactory.hpp"
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Stratimikos_MueLuHelpers.hpp"

namespace Albany
{

  //Decorator of Thyra::LinearOpWithSolveBase that provides a method to initialize the solver.
  class Init_LOWS : public Thyra_LOWS
  {
  public:
    virtual void initializeSolver(Teuchos::RCP<Teuchos::ParameterList> solverParamList) = 0;
  };


  //! MatrixBased_LOWS provides a concrete implementation of LinearOpWithSolve based on an existing matrix
  /*!
    * This class imports a given matrix (linear operator) and allows to initialize the solver
    * using a provided Stratimikos parameter list.
    */
  class MatrixBased_LOWS : public Init_LOWS
  {
  public:
    // Constructor
    MatrixBased_LOWS(
        const Teuchos::RCP<Thyra_LinearOp> &matrix);

    //! Destructor
    virtual ~MatrixBased_LOWS();

    //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
    Teuchos::RCP<const Thyra_VectorSpace> domain() const;

    //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
    Teuchos::RCP<const Thyra_VectorSpace> range() const;

    //! Returns the matrix (linear operator) passed to the constructor
    Teuchos::RCP<Thyra_LinearOp> getMatrix();

    //! Initilializes the solver from a parameter list with Stratimikos parameters  
    void initializeSolver(Teuchos::RCP<Teuchos::ParameterList> solverParamList);

    //@}

  protected:
    //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
    bool opSupportedImpl(Thyra::EOpTransp M_trans) const;

    //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
    void applyImpl(const Thyra::EOpTransp M_trans,
                   const Thyra_MultiVector &X,
                   const Teuchos::Ptr<Thyra_MultiVector> &Y,
                   const ST alpha,
                   const ST beta) const;

    //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
    Thyra::SolveStatus<double> solveImpl(
        const Thyra::EOpTransp transp,
        const Thyra_MultiVector &B,
        const Teuchos::Ptr<Thyra_MultiVector> &X,
        const Teuchos::Ptr<const Thyra::SolveCriteria<ST>> solveCriteria) const;


    //! stored the matrix passed to the constructor
    const Teuchos::RCP<Thyra_LinearOp> mat_;

    //! The Thyra LinearOpWithSolve object
    Teuchos::RCP<Thyra_LOWS> solver_;
    Teuchos::RCP<Thyra_LOWS> solver_transp_;
    //@}

  }; // class MatrixBased_LOWS

} // namespace Albany

#endif // ALBANY_LINEAROPWITHSOLVEDECORATORS_HPP
