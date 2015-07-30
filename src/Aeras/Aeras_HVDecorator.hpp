//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Aeras_HVDecorator_hpp)
#define Aeras_HVDecorator_hpp

#include "Albany_ModelEvaluatorT.hpp"
#include "Albany_DataTypes.hpp"
#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"

#include "Thyra_ModelEvaluatorDefaultBase.hpp"

namespace Aeras {

///
/// \brief Definition for the HyperViscosityDecorator
///
class HVDecorator: public Albany::ModelEvaluatorT {

public:

  /// Constructor
  HVDecorator(
      const Teuchos::RCP<Albany::Application>& app,
      const Teuchos::RCP<Teuchos::ParameterList>& appParams);

  Teuchos::RCP<Tpetra_CrsMatrix> createOperator(double alpha, double beta, double omega); 

protected:

  //! Evaluate model on InArgs
  void evalModelImpl(
      const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgs,
      const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgs) const;

private: 
  //Mass and Laplace operators
  Teuchos::RCP<Tpetra_CrsMatrix> mass_;  
  Teuchos::RCP<Tpetra_CrsMatrix> laplace_;  

};

}

#endif // ALBANY_HVDECORATOR_HPP
