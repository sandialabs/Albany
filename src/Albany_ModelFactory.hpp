//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra ifdef'ed out!

#ifndef ALBANY_MODELFACTORY_HPP
#define ALBANY_MODELFACTORY_HPP

#if defined(ALBANY_EPETRA)
#include "EpetraExt_ModelEvaluator.h"
#endif
#include "Thyra_ModelEvaluatorDefaultBase.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_DataTypes.hpp"

namespace Albany {

class Application;

class ModelFactory {
public:
  ModelFactory(const Teuchos::RCP<Teuchos::ParameterList> &params,
               const Teuchos::RCP<Application> &app);

#if defined(ALBANY_EPETRA)
  Teuchos::RCP<EpetraExt::ModelEvaluator> create() const;
#endif
  //Thyra version of above
  Teuchos::RCP<Thyra::ModelEvaluatorDefaultBase<ST> > createT() const;

private:
  Teuchos::RCP<Teuchos::ParameterList> params_;
  Teuchos::RCP<Application> app_;

  // Disallow copy & assignment
  ModelFactory(const ModelFactory &);
  ModelFactory &operator=(const ModelFactory &);
};

} // end namespace Albany

#endif /* ALBANY_MODELFACTORY_HPP */
