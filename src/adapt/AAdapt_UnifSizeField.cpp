//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_UnifSizeField.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"
#include "Epetra_Import.h"

AAdapt::UnifSizeField::UnifSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc) :
  comm(disc->getComm()) {
}

AAdapt::UnifSizeField::
~UnifSizeField() {
}

void
AAdapt::UnifSizeField::computeError() {
}


void
AAdapt::UnifSizeField::setParams(
				 double element_size, double err_bound,
				 const std::string state_var_name) {

  elem_size = element_size;

}

double AAdapt::UnifSizeField::getValue(ma::Entity* v) {
  return elem_size;
}

