//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_NonUnifRefSizeField.hpp"
#include "Albany_PUMIMeshStruct.hpp"

#include "Albany_Utils.hpp"

AAdapt::NonUnifRefSizeField::NonUnifRefSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc) :
  MeshSizeField(disc) {
}

AAdapt::NonUnifRefSizeField::
~NonUnifRefSizeField() {
}

void
AAdapt::NonUnifRefSizeField::configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) {

     int num_iters = adapt_params_->get<int>("Max Number of Mesh Adapt Iterations", 1);

     ma::Input *in = ma::configureUniformRefine(mesh_struct->getMesh(), num_iters);
     in->maximumIterations = num_iters;

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

     in->shouldFixShape = true;

     ma::adapt(in);

}

void
AAdapt::NonUnifRefSizeField::computeError() {
}

void
AAdapt::NonUnifRefSizeField::setParams(
    const Teuchos::RCP<Teuchos::ParameterList>& p) {

}

void
AAdapt::NonUnifRefSizeField::copyInputFields() {

}

