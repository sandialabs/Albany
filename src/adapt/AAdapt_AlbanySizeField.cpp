//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_AlbanySizeField.hpp"
#include "Albany_PUMIMeshStruct.hpp"

#include "Albany_Utils.hpp"

AAdapt::AlbanySizeField::AlbanySizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc) :
  MeshSizeField(disc) {
}

AAdapt::AlbanySizeField::
~AlbanySizeField() {
}

void
AAdapt::AlbanySizeField::configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_)
{

  apf::Field* field = mesh_struct->getMesh()->findField("IsoMeshSizeField");
  TEUCHOS_TEST_FOR_EXCEPTION(field == NULL, std::logic_error, "Cannot find IsoMeshSizeField");

  ma::Input *in = ma::configure(mesh_struct->getMesh(), field);

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

/*
double AAdapt::AlbanySizeField::getValue(ma::Entity* v) {

  // "field" is the L2 projected Albany size field calculated in an Albany evaluator.
  // It is projected by "Project IP to Nodal Field" and named "IsoMeshSizeField" in the vtk output file

  return apf::getScalar(field,v,0);

}
*/

