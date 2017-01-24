//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_UniformRefine.hpp"
#include "Albany_PUMIMeshStruct.hpp"

#include "Albany_Utils.hpp"

AAdapt::UniformRefine::UniformRefine(const Teuchos::RCP<Albany::APFDiscretization>& disc) :
  MeshAdaptMethod(disc) {
}

AAdapt::UniformRefine::
~UniformRefine() {
}

void
AAdapt::UniformRefine::adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) {

     int num_iters = adapt_params_->get<int>("Max Number of Mesh Adapt Iterations", 1);

     ma::Input *in = ma::configureUniformRefine(mesh_struct->getMesh(), num_iters);
     in->maximumIterations = num_iters;

     //do not snap on deformation problems even if the model supports it
     in->shouldSnap = false;

     setCommonMeshAdaptOptions(adapt_params_, in);

     in->shouldFixShape = true;

     ma::adapt(in);

}

void
AAdapt::UniformRefine::preProcessShrunkenMesh() {
}

void
AAdapt::UniformRefine::setParams(
    const Teuchos::RCP<Teuchos::ParameterList>& p) {

}

void
AAdapt::UniformRefine::preProcessOriginalMesh() {

}

