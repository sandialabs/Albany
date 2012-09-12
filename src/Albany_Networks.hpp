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

