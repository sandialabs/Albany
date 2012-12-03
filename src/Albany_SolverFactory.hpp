//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLVERFACTORY_HPP
#define ALBANY_SOLVERFACTORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "EpetraExt_ModelEvaluator.h"
#include "Teuchos_SerialDenseVector.hpp"
#include "Epetra_Vector.h"
#include "Thyra_VectorBase.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Albany_Utils.hpp"

#include "Albany_Application.hpp"

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

    //! Destructor
    virtual ~SolverFactory() {}

    //! Create solver as response-only model evaluator
    virtual Teuchos::RCP<EpetraExt::ModelEvaluator> create(
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Comm>& solverComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess = Teuchos::null);

    Teuchos::RCP<EpetraExt::ModelEvaluator> createAndGetAlbanyApp(
      Teuchos::RCP<Application>& albanyApp,
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Comm>& solverComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null);

    Teuchos::RCP<EpetraExt::ModelEvaluator> createAlbanyAppAndModel(
      Teuchos::RCP<Application>& albanyApp,
      const Teuchos::RCP<const Epetra_Comm>& appComm,
      const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null);

    /** \brief Function that does regression testing. */
    // Probably needs to be moved to another class? AGS
    int checkTestResults(
      int response_index,
      int parameter_index,
      const Epetra_Vector* g,
      const Epetra_MultiVector* dgdp,
      const Teuchos::SerialDenseVector<int,double>* drdv = NULL,
      const Teuchos::RCP<Thyra::VectorBase<double> >& tvec = Teuchos::null,
      const Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly>& g_sg = Teuchos::null,
      const Epetra_Vector* g_mean = NULL,
      const Epetra_Vector* g_std_dev = NULL) const;

    Teuchos::ParameterList& getAnalysisParameters() const
      { return appParams->sublist("Piro").sublist("Analysis"); }

    Teuchos::ParameterList& getParameters() const
      { return *appParams; }

    //! Set rigid body modes in parameter list
    void setRigidBodyModesForML(Teuchos::ParameterList& mlList,
				Albany::Application& app);

    //! Function to get coodinates from the mesh and insert to ML precond list
    void setCoordinatesForML(const string& solutionMethod, 
                    const string& secondOrder,
                    Teuchos::RCP<Teuchos::ParameterList>& piroParams,
                    Teuchos::RCP<Albany::Application>& app,
                    std::string& problemName);

  private:

    // Private functions to set deafult parameter values
    void setSolverParamDefaults(Teuchos::ParameterList* appParams, int myRank);

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
    
    Teuchos::RCP<Teuchos::FancyOStream> out;

  };

}

#endif
