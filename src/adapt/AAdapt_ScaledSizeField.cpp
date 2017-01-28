//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_ScaledSizeField.hpp"
#include "Albany_PUMIMeshStruct.hpp"

#include "Albany_Utils.hpp"

AAdapt::ScaledSizeField::ScaledSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc) :
  MeshAdaptMethod(disc) {
}

void
AAdapt::ScaledSizeField::adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_)
{

  ma::IsotropicFunction*
    isf = dynamic_cast<ma::IsotropicFunction*>(&scaledIsoFunc);
  ma::Input *in = ma::configure(mesh_struct->getMesh(), isf);

  in->maximumIterations = adapt_params_->get<int>("Max Number of Mesh Adapt Iterations", 1);
  //do not snap on deformation problems even if the model supports it
  in->shouldSnap = false;

  setCommonMeshAdaptOptions(adapt_params_, in);

  ma::adapt(in);

}

void
AAdapt::ScaledSizeField::preProcessOriginalMesh()
{

  scaledIsoFunc.averageEdgeLength_ = ma::getAverageEdgeLength(mesh_struct->getMesh());

}

AAdapt::ScaledSizeField::
~ScaledSizeField() {
}

void
AAdapt::ScaledSizeField::preProcessShrunkenMesh() {
}

void
AAdapt::ScaledSizeField::setParams(
    const Teuchos::RCP<Teuchos::ParameterList>& p) {

  scaledIsoFunc.factor_ = p->get<double>("Element Size Scaling", 0.7);

}

