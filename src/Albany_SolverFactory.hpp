//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLVERFACTORY_HPP
#define ALBANY_SOLVERFACTORY_HPP

#include "Albany_Utils.hpp"
#include "Albany_Application.hpp"

#include "Thyra_ModelEvaluator.hpp"
#include "Thyra_VectorBase.hpp"

#include "EpetraExt_ModelEvaluator.h"
#include "Thyra_ModelEvaluator.hpp"
#include "Thyra_ResponseOnlyModelEvaluatorBase.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Epetra_Vector.h"

#include "Stokhos_EpetraVectorOrthogPoly.hpp"

#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "AAdapt_AdaptiveModelFactory.hpp"


//! Albany driver code, problems, discretizations, and responses
namespace Albany {

  /*!
   * \brief A factory class to instantiate AbstractSolver objects
   */
  class SolverFactory {
  public:

    //! Default constructor
    SolverFactory(const std::string& inputfile,
		  const Albany_MPI_Comm& mcomm);

    SolverFactory(const Teuchos::RCP<Teuchos::ParameterList>& input_appParams,
		  const Albany_MPI_Comm& mcomm);


    //! Destructor
    virtual ~SolverFactory();

    //! Create solver as response-only model evaluator
    virtual Teuchos::RCP<EpetraExt::ModelEvaluator> create(
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Comm>& solverComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess = Teuchos::null);

   // Thyra version of above
   virtual Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> > createT(
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Comm>& solverComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess = Teuchos::null);

    Teuchos::RCP<EpetraExt::ModelEvaluator> createAndGetAlbanyApp(
      Teuchos::RCP<Application>& albanyApp,
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Comm>& solverComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null);

    //Thyra version of above
    Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST> > createAndGetAlbanyAppT(
      Teuchos::RCP<Application>& albanyApp,
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Comm>& solverComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null);

    Teuchos::RCP<Thyra::ModelEvaluator<double> > createThyraSolverAndGetAlbanyApp(
      Teuchos::RCP<Application>& albanyApp,
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Comm>& solverComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null);

    Teuchos::RCP<EpetraExt::ModelEvaluator> createAlbanyAppAndModel(
      Teuchos::RCP<Application>& albanyApp,
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null);

    //Thyra version of above
    Teuchos::RCP<Thyra::ModelEvaluator<ST> > createAlbanyAppAndModelT(
      Teuchos::RCP<Application>& albanyApp,
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null);

    Teuchos::ParameterList& getAnalysisParameters() const
      { return appParams->sublist("Piro").sublist("Analysis"); }

    Teuchos::ParameterList& getParameters() const
      { return *appParams; }


  public:
    
    // Functions to generate reference parameter lists for validation
    //  EGN 9/2013: made these three functions public, as they pertain to valid 
    //    parameter lists for Albany::Application objects, which may get created
    //    apart from Albany::SolverFactory.  It may be better to relocate these 
    //    to the Application class, or as functions "related to" Albany::Application.
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidAppParameters() const;
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidParameterParameters() const;
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidResponseParameters() const;

 
  private:

    // Private functions to set default parameter values
    void setSolverParamDefaults(Teuchos::ParameterList* appParams, int myRank);

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidRegressionResultsParameters() const;

    //! Private to prohibit copying
    SolverFactory(const SolverFactory&);

    //! Private to prohibit copying
    SolverFactory& operator=(const SolverFactory&);

  public:
    /** \brief Function that does regression testing for problem solves. */
    int checkSolveTestResults(
      int response_index,
      int parameter_index,
      const Epetra_Vector* g,
      const Epetra_MultiVector* dgdp) const;

    /** \brief Function that does regression testing for Dakota runs. */
    int checkDakotaTestResults(
      int response_index,
      const Teuchos::SerialDenseVector<int,double>* drdv) const;

    /** \brief Function that does regression testing for Analysis runs. */
    int checkAnalysisTestResults(
      int response_index,
      const Teuchos::RCP<Thyra::VectorBase<double> >& tvec) const;

    /** \brief Function that does regression testing for SG runs. */
    int checkSGTestResults(
      int response_index,
      const Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& g_sg,
      const Epetra_Vector* g_mean = NULL,
      const Epetra_Vector* g_std_dev = NULL) const;

  private:
    /** \brief Testing utility that compares two numbers using two tolerances */
    int scaledCompare(double x1, double x2, double relTol, double absTol) const;

    Teuchos::ParameterList *getTestParameters(int response_index) const;

    void storeTestResults(
        Teuchos::ParameterList* testParams,
        int failures,
        int comparisons) const;

  protected:
    //! Parameter list specifying what solver to create
    Teuchos::RCP<Teuchos::ParameterList> appParams;

    Teuchos::RCP<Teuchos::FancyOStream> out;

    Teuchos::RCP<AAdapt::AdaptiveModelFactory> thyraModelFactory;
  };

}

#endif
