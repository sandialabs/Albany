/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_SOLVERFACTORY_HPP
#define ALBANY_SOLVERFACTORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "EpetraExt_ModelEvaluator.h"
#include "Epetra_Vector.h"
#include "Teuchos_TestForException.hpp"
#include "Thyra_VectorBase.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"

// This is needed for autoconf builds
#ifdef HAVE_MPI 
 #ifndef ALBANY_MPI
  #define ALBANY_MPI
 #endif
#endif

#ifdef ALBANY_MPI
#include "Epetra_MpiComm.h"
#else
typedef int MPI_Comm;
#define MPI_COMM_WORLD 1
#include "Epetra_SerialComm.h"
#endif

#include "Rythmos_IntegrationObserverBase.hpp"
#include "Albany_Application.hpp"

#include "NOX_Epetra_Observer.H"

namespace Albany {

  /*!
   * \brief A factory class to instantiate AbstractSolver objects
   */
  class SolverFactory {
  public:

    //! Default constructor
    SolverFactory(const std::string inputfile,
		  const Teuchos::RCP<const Epetra_Comm>& comm);

    //! Destructor
    virtual ~SolverFactory() {}

    //! Create model evaluator for this problem
    /*!
     * If \c appComm is null, then the comm created within this class
     * will be used.
     */
    virtual void createModel(
      const Teuchos::RCP<const Epetra_Comm>& appComm = Teuchos::null,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess = Teuchos::null);

    //! Create solver as response-only model evaluator
    virtual Teuchos::RCP<EpetraExt::ModelEvaluator> create();

    //! Get application
    Teuchos::RCP<Albany::Application> getApplication() { return app; }

    /** \brief Function that does regression testing. */
    // Probably needs to be moved to another class? AGS
    int checkTestResults(
      const Epetra_Vector* g,
      const Epetra_MultiVector* dgdp,
      const Teuchos::SerialDenseVector<int,double>* drdv = NULL,
      const Teuchos::RCP<Thyra::VectorBase<double> >& tvec = Teuchos::null,
      const Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& g_sg = Teuchos::null) const;

    Teuchos::ParameterList& getAnalysisParameters() const
      { return appParams->sublist("Analysis"); }

    Teuchos::ParameterList& getParameters() const
      { return *appParams; }

  private:

    // Private functions to set deafult parameter values
    void setSolverParamDefaults(Teuchos::ParameterList* appParams);

    // Functions to generate reference parameter lists for validation
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidAppParameters() const;
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidRegressionResultsParameters() const;
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidParameterParameters() const;
    Teuchos::RCP<const Teuchos::ParameterList>
      getValidResponseParameters() const;

    //! Private to prohibit copying
    SolverFactory(const SolverFactory&);

    //! Private to prohibit copying
    SolverFactory& operator=(const SolverFactory&);

    /** \brief Testing utility that compares two numbers using two tolerances */
    int scaledCompare(double x1, double x2, double relTol, double absTol) const;

  protected:

    //! Parameter list specifying what solver to create
    Teuchos::RCP<Teuchos::ParameterList> appParams;
    Teuchos::RCP<const Epetra_Comm> Comm;
    Teuchos::RCP<Albany::Application> app;
    Teuchos::RCP<EpetraExt::ModelEvaluator> model;

    bool transient;
    bool continuation;
    bool stochastic;
    int numParameters;

    typedef double Scalar;
    Teuchos::RCP<Rythmos::IntegrationObserverBase<Scalar> > Rythmos_observer;
    Teuchos::RCP<NOX::Epetra::Observer > NOX_observer;
    Teuchos::RCP<Teuchos::FancyOStream> out;

  };

}

#endif
