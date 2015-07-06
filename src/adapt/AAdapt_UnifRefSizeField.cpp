//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_UnifRefSizeField.hpp"
#include "Albany_PUMIMeshStruct.hpp"

#include "Albany_Utils.hpp"

AAdapt::UnifRefSizeField::UnifRefSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc) :
  MeshSizeField(disc) {
}

void
AAdapt::UnifRefSizeField::configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_)
{

  ma::IsotropicFunction*
    isf = dynamic_cast<ma::IsotropicFunction*>(&unifRefIsoFunc);
  ma::Input *in = ma::configure(mesh_struct->getMesh(), isf);

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

  ma::adapt(in);

}

void
AAdapt::UnifRefSizeField::copyInputFields()
{

  unifRefIsoFunc.averageEdgeLength = ma::getAverageEdgeLength(mesh_struct->getMesh());

}

AAdapt::UnifRefSizeField::
~UnifRefSizeField() {
}

void
AAdapt::UnifRefSizeField::computeError() {
}

void
AAdapt::UnifRefSizeField::setParams(
    const Teuchos::RCP<Teuchos::ParameterList>& p) {

  unifRefIsoFunc.elem_size = p->get<double>("Target Element Size", 0.7);

}

