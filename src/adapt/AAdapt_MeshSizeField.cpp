//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_MeshSizeField.hpp"

namespace AAdapt {

MeshSizeField::MeshSizeField(
    const Teuchos::RCP<Albany::APFDiscretization>& disc): 
    mesh_struct(disc->getAPFMeshStruct()),
    commT(disc->getComm())
{
}

ma::Input*
MeshSizeField::configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_)
{
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

}
