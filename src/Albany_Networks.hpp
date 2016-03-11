//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: right now this is Epetra (Albany) function.
//Not compiled if ALBANY_EPETRA_EXE is off.

#ifndef ALBANY_NETWORKS_HPP
#define ALBANY_NETWORKS_HPP

#include "Piro_Epetra_NECoupledModelEvaluator.hpp"

namespace Albany {

  class ReactorNetworkModel : public Piro::Epetra::AbstractNetworkModel {
  
  public:
    
    //! Constructor
    ReactorNetworkModel(int n_) : n(n_) {}
    
    //! Destructor
    virtual ~ReactorNetworkModel() {}
    
    //! evaluate model
    virtual void evalModel(
      const Teuchos::Array<EpetraExt::ModelEvaluator::InArgs>& model_inargs, 
      const Teuchos::Array<EpetraExt::ModelEvaluator::OutArgs>& model_outargs,
      const EpetraExt::ModelEvaluator::InArgs& network_inargs, 
      const EpetraExt::ModelEvaluator::OutArgs& network_outargs,
      const Teuchos::Array<int>& n_p,
      const Teuchos::Array<int>& n_g,
      const Teuchos::Array< Teuchos::RCP<Epetra_Vector> >& p,
      const Teuchos::Array< Teuchos::RCP<Epetra_Vector> >& g,
      const Teuchos::Array< Teuchos::RCP<Epetra_MultiVector> >& dgdp,
      const Teuchos::Array<EpetraExt::ModelEvaluator::EDerivativeMultiVectorOrientation>& dgdp_layout,
      const Teuchos::Array<EpetraExt::ModelEvaluator::OutArgs::sg_vector_t>& p_sg,
      const Teuchos::Array<EpetraExt::ModelEvaluator::OutArgs::sg_vector_t>& g_sg,
      const Teuchos::Array<Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> >& dgdp_sg,
      const Teuchos::Array<EpetraExt::ModelEvaluator::EDerivativeMultiVectorOrientation>& dgdp_sg_layout) const;
    
  protected:
    int n;
  };

}

#endif

