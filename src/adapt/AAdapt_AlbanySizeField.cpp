//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_AlbanySizeField.hpp"
#include "Albany_PUMIMeshStruct.hpp"

#include "Albany_Utils.hpp"

AAdapt::AlbanySizeField::AlbanySizeField(const Teuchos::RCP<Albany::AbstractPUMIDiscretization>& disc) :
  MeshSizeField(disc) {
}

void
AAdapt::AlbanySizeField::copyInputFields()
{

  averageEdgeLength = ma::getAverageEdgeLength(mesh_struct->getMesh());

}

AAdapt::AlbanySizeField::
~AlbanySizeField() {
}

void
AAdapt::AlbanySizeField::computeError() {
}

void
AAdapt::AlbanySizeField::setParams(
    const Teuchos::RCP<Teuchos::ParameterList>& p) {

  elem_size = p->get<double>("Target Element Size", 0.7);

}

double AAdapt::AlbanySizeField::getValue(ma::Entity* v) {
  return elem_size * averageEdgeLength;
}

