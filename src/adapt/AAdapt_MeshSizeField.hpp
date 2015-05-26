//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_MESHSIZEFIELD_HPP
#define AADAPT_MESHSIZEFIELD_HPP

namespace AAdapt {
/*! \brief Define methods that Albany wants a size field to implement in
 *         addition to those in ma::SizeField.
 */
struct MeshSizeField {

  MeshSizeField(const Teuchos::RCP<Albany::AbstractPUMIDiscretization>& disc) : 
    mesh_struct(disc->getPUMIMeshStruct()), commT(disc->getComm()) {}

  virtual ma::Input* configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_){

     ma::Input *in = NULL;

     ma::IsotropicFunction*
       isf = dynamic_cast<ma::IsotropicFunction*>(this);
     if (isf) in = ma::configure(mesh_struct->getMesh(), isf);

     ma::AnisotropicFunction*
       asf = dynamic_cast<ma::AnisotropicFunction*>(this);
     if (asf) in = ma::configure(mesh_struct->getMesh(), asf);

     TEUCHOS_TEST_FOR_EXCEPTION(!in, std::logic_error, "shouldn't be here");

     in->maximumIterations = adapt_params_->get<int>("Max Number of Mesh Adapt Iterations", 1);
     //do not snap on deformation problems even if the model supports it
     in->shouldSnap = false;

     bool loadBalancing = adapt_params_->get<bool>("Load Balancing",true);
     double lbMaxImbalance = adapt_params_->get<double>("Maximum LB Imbalance",1.30);
     if (loadBalancing) {
       in->shouldRunPreZoltan = true;
       in->shouldRunMidParma = true;
       in->shouldRunPostParma = true;
       in->maximumImbalance = lbMaxImbalance;
     }
     return in;
  }

  virtual void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p) = 0;
  virtual void computeError() = 0;
  virtual void copyInputFields() = 0;
  virtual void freeInputFields() = 0;
  virtual void freeSizeField() = 0;

protected:

  Teuchos::RCP<Albany::PUMIMeshStruct> mesh_struct;
  Teuchos::RCP<const Teuchos_Comm> commT;

};
}

#endif
