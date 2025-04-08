//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_QUADRATICLINEAROPERATORBASEDRESPONSEFUNCTION_HPP
#define ALBANY_QUADRATICLINEAROPERATORBASEDRESPONSEFUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"
#include "Albany_LinearOpWithSolveDecorators.hpp"
#include "Albany_Utils.hpp"

namespace Albany {
  class Application;
  class AtDinvA_LOWS;

  /*!
   * \brief Response function computing the scalar:
   * coeff p' A' inv(D) A p,
   * for a field parameter p, a matrix A and a invertible symmetric matrix D.
   * The matrices A and D are loaded from "*.mm" ASCII files.
   * Performance optimizations implemented when A is symmetric, D diagonal or
   * when A is equal to D. 
   * When A is equal to D, the response reduces to coeff p'A p
   */
  class QuadraticLinearOperatorBasedResponseFunction :
    public SamplingBasedScalarResponseFunction {
  public:
  
    //! Default constructor
    QuadraticLinearOperatorBasedResponseFunction(
        const Teuchos::RCP<const Application> &app,
        const Teuchos::ParameterList &responseParams);

    //! Destructor
    virtual ~QuadraticLinearOperatorBasedResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    void loadLinearOperator();

    //! Evaluate responses
    virtual void 
    evaluateResponse(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
      const Teuchos::RCP<Thyra_Vector>& g);


    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangent(const double alpha, 
      const double beta,
      const double omega,
      const double current_time,
      bool sum_derivs,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      Teuchos::Array<ParamVec>& p,
      const int parameter_index,
      const Teuchos::RCP<const Thyra_MultiVector>& Vx,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vp,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& gx,
      const Teuchos::RCP<Thyra_MultiVector>& gp);
    
    virtual void 
    evaluateGradient(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& p,
      const int parameter_index,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

    //! returns Hessian Linear Operator
    virtual Teuchos::RCP<Thyra_LinearOp>
    get_Hess_pp_operator(const std::string& param_name);

  void printResponse(Teuchos::RCP<Teuchos::FancyOStream> out);

  private:

    //! Evaluate distributed parameter derivative = dg/dp
    virtual void
    evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

    virtual void
    evaluate_HessVecProd_xx(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

    virtual void
    evaluate_HessVecProd_xp(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

    virtual void
    evaluate_HessVecProd_px(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

    virtual void
    evaluate_HessVecProd_pp(
      const double current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const std::string& dist_param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>& Hv_dp);

  private:

    //! Private to prohibit copying
    QuadraticLinearOperatorBasedResponseFunction(const QuadraticLinearOperatorBasedResponseFunction&);
    
    QuadraticLinearOperatorBasedResponseFunction& operator=(const QuadraticLinearOperatorBasedResponseFunction&);

    const Teuchos::RCP<const Application> app_;
    std::string field_name_;
    std::string target_name_;
    Teuchos::RCP<AtDinvA_LOWS> twoAtDinvA_;
    Teuchos::RCP<Thyra_Vector> g_;
  };



    //! Concrete implementation of Thyra::LinearOpWithSolveBase, for the operator coeff A' D^{-1} A
      /*!
      * A is a sparse matrix, and D is an invertible symmetric sparse matrix, and coeff a scalar coefficient
      * A and D are imported from files, they have same range and domain space, which is passed using the method setupFwdOp
      * Performance optimizations implemented when A is symmetric, D diagonal or
      * when A is equal to D. 
      * When A is equal to D, the operator reduces to coeff A
      */
    class AtDinvA_LOWS : public Init_LOWS {
    public:

      // Constructor
      AtDinvA_LOWS(
        const std::string& file_name_A,
        const std::string& file_name_D,
        const double& coeff,
        Teuchos::RCP<Teuchos::ParameterList> solverParamList,
        const bool symmetricA = false,
        const bool diagonalD = false);

      //! Destructor
      virtual ~AtDinvA_LOWS();


      //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
      Teuchos::RCP<const Thyra_VectorSpace> domain() const;

      //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
      Teuchos::RCP<const Thyra_VectorSpace> range() const;

      //! Imports the Thyra VectorSpace and uses it to define the range and domain of A.
      //! The matrices A and D are then imported from file.
      //! If the matrix A and D have been already created, this function returns without performing any operation
      void setupFwdOp(const Teuchos::RCP<const Thyra_VectorSpace>& vec_space);

      //  coeff X' A' inv(D) A  X
      ST quadraticForm(const Thyra_MultiVector& X);

      //! Initilializes the solver from a parameter list with Stratimikos parameters  
      void initializeSolver(Teuchos::RCP<Teuchos::ParameterList> solverParamList);
      
      //! Initilializes the solver for matrix D from a parameter list with Stratimikos parameters  
      void initializeFwdSolver();


    protected:
      //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
      bool opSupportedImpl(Thyra::EOpTransp /*M_trans*/) const;

      //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
      void applyImpl (const Thyra::EOpTransp /*M_trans*/, //operator is symmetric by construction
                      const Thyra_MultiVector& X,
                      const Teuchos::Ptr<Thyra_MultiVector>& Y,
                      const ST alpha,
                      const ST beta) const;

      //! Overrides Thyra::LinearOpWithSolveBase purely virtual method
      Thyra::SolveStatus<double> solveImpl(
        const Thyra::EOpTransp transp,
        const Thyra_MultiVector &B,
        const Teuchos::Ptr<Thyra_MultiVector> &X,
        const Teuchos::Ptr<const Thyra::SolveCriteria<ST> > solveCriteria
        ) const;

    private:
      //! sets the range and domain of the matrices A and D and imports them from file
      void loadLinearOperators();

      //! files storing the operators A and D
      std::string file_name_A_, file_name_D_;

      //! scaling coefficient
      double coeff_; 

      //! matrices A and D
      Teuchos::RCP<Thyra_LinearOp> A_, D_;

      //! solvers for A and A' and D
      Teuchos::RCP<Thyra_LOWS> A_solver_, A_transSolver_, D_solver_;

      //! vector space which is also the range and domain of A
      Teuchos::RCP<const Thyra_VectorSpace> vec_space_;

      //! internal auxiliary vectors for computations
      Teuchos::RCP<Thyra_Vector> vecD_,vec1_,vec2_;

      //! Builder for linear solvers using Stratimikos parameter lists
      Stratimikos::DefaultLinearSolverBuilder fwdLinearSolverBuilder_;

      //! booleans denoting whether matrix A is symmetric, D is diagonal or A equals D
      bool symmetricA_, diagonalD_, AequalsD_;


    }; // class DistributedParameterDerivativeOp

} // namespace Albany

#endif // ALBANY_SOLUTIONTWONORMRESPONSEFUNCTION_HPP
