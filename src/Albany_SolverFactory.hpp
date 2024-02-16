//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_SOLVER_FACTORY_HPP
#define ALBANY_SOLVER_FACTORY_HPP

#include "Albany_Application.hpp"
#include "Albany_ModelEvaluator.hpp"

#include "Thyra_ModelEvaluator.hpp"
#include "Thyra_VectorBase.hpp"

#include "Piro_ObserverBase.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Thyra_ResponseOnlyModelEvaluatorBase.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseVector.hpp"

//! Albany driver code, problems, discretizations, and responses
namespace Albany {

/*!
 * \brief A factory class to instantiate AbstractSolver objects
 */
class SolverFactory {
public:
  //! Default constructor
  SolverFactory(
      const std::string&                      inputfile,
      const Teuchos::RCP<const Teuchos_Comm>& comm);

  SolverFactory(
      const Teuchos::RCP<Teuchos::ParameterList>& input_appParams,
      const Teuchos::RCP<const Teuchos_Comm>&     comm);

  //! Destructor
  virtual ~SolverFactory () = default;

  Teuchos::RCP<Application>
  createApplication (const Teuchos::RCP<const Teuchos_Comm>&  appComm,
                     const Teuchos::RCP<const Thyra_Vector>& initial_guess = Teuchos::null);

  Teuchos::RCP<ModelEvaluator>
  createModel (const Teuchos::RCP<Application>& app,
	       const bool adjoint_model = false);

  Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<ST>>
  createSolver (const Teuchos::RCP<const Teuchos_Comm>&    solverComm, 
                const Teuchos::RCP<ModelEvaluator>&  model,
		const Teuchos::RCP<ModelEvaluator>&  adjointModel,
                const bool forwardMode); 

  Teuchos::ParameterList&
  getAnalysisParameters() const
  {
    return m_appParams->sublist("Piro").sublist("Analysis");
  }

  Teuchos::RCP<Teuchos::ParameterList>
  getParameters() const
  {
    return m_appParams;
  }

  Teuchos::RCP<Piro::ObserverBase<double>>
  returnObserver() const
  {
    return m_observer;
  };

  // Functions to generate reference parameter lists for validation
  //  EGN 9/2013: made these three functions public, as they pertain to valid
  //    parameter lists for Albany::Application objects, which may get created
  //    apart from Albany::SolverFactory.  It may be better to relocate these
  //    to the Application class, or as functions "related to"
  //    Albany::Application.
  Teuchos::RCP<const Teuchos::ParameterList>  getValidAppParameters()       const;
  Teuchos::RCP<const Teuchos::ParameterList>  getValidDebugParameters()     const; 
  Teuchos::RCP<const Teuchos::ParameterList>  getValidScalingParameters()   const; 
  Teuchos::RCP<const Teuchos::ParameterList>  getValidParameterParameters() const;
  Teuchos::RCP<const Teuchos::ParameterList>  getValidResponseParameters()  const;

protected:

  void setup (const Teuchos::RCP<Teuchos::ParameterList>& input_appParams,
              const Teuchos::RCP<const Teuchos_Comm>&     comm);

  void setSolverParamDefaults(Teuchos::ParameterList* appParams, int myRank);

  Teuchos::RCP<Piro::ObserverBase<double>>  m_observer;

  //! Parameter list specifying what solver to create
  Teuchos::RCP<Teuchos::ParameterList>      m_appParams;

  Teuchos::RCP<Teuchos::FancyOStream>       m_out;
};

}  // namespace Albany

#endif // ALBANY_SOLVER_FACTORY_HPP
