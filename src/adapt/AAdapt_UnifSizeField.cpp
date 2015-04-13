//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_UnifSizeField.hpp"
#include "Albany_PUMIMeshStruct.hpp"

AAdapt::UnifSizeField::UnifSizeField(const Teuchos::RCP<Albany::AbstractPUMIDiscretization>& disc) :
  commT(disc->getComm()) {
}

AAdapt::UnifSizeField::
~UnifSizeField() {
}

void
AAdapt::UnifSizeField::computeError() {
}


void
AAdapt::UnifSizeField::setParams(
    const Teuchos::RCP<Teuchos::ParameterList>& p) {
  elem_size = p->get<double>("Target Element Size", 0.1);
}

double AAdapt::UnifSizeField::getValue(ma::Entity* v) {
  return elem_size;
}

