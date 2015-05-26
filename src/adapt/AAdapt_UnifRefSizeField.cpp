//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_UnifRefSizeField.hpp"
#include "Albany_PUMIMeshStruct.hpp"

#include "Albany_Utils.hpp"

AAdapt::UnifRefSizeField::UnifRefSizeField(const Teuchos::RCP<Albany::AbstractPUMIDiscretization>& disc) :
  MeshSizeField(disc) {

  initialAverageEdgeLength = ma::getAverageEdgeLength(mesh_struct->getMesh());

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

  elem_size = p->get<double>("Target Element Size", 0.1);

}

double AAdapt::UnifRefSizeField::getValue(ma::Entity* v) {
  return 0.5 * initialAverageEdgeLength;
}

