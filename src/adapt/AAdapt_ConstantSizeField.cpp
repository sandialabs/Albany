//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_ConstantSizeField.hpp"
#include "Albany_PUMIMeshStruct.hpp"

AAdapt::ConstantSizeField::ConstantSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc) :
  MeshAdaptMethod(disc) {
}

AAdapt::ConstantSizeField::
~ConstantSizeField() {
}

void
AAdapt::ConstantSizeField::adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_)
{

  ma::IsotropicFunction*
    isf = dynamic_cast<ma::IsotropicFunction*>(&constantIsoFunc);
  ma::Input *in = ma::configure(mesh_struct->getMesh(), isf);

  in->maximumIterations = adapt_params_->get<int>("Max Number of Mesh Adapt Iterations", 1);
  //do not snap on deformation problems even if the model supports it
  in->shouldSnap = false;

  setCommonMeshAdaptOptions(adapt_params_, in);

  ma::adapt(in);

}

void
AAdapt::ConstantSizeField::preProcessShrunkenMesh() {
}


void
AAdapt::ConstantSizeField::setParams(
    const Teuchos::RCP<Teuchos::ParameterList>& p) {
  constantIsoFunc.value_ = p->get<double>("Target Element Size", 0.1);
}

