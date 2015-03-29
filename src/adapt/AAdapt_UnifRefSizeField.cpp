//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "AAdapt_UnifRefSizeField.hpp"
#include "AlbPUMI_FMDBMeshStruct.hpp"

#include "Albany_Utils.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/all_reduce.hpp>

AAdapt::UnifRefSizeField::UnifRefSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc) :
  mesh(disc->getFMDBMeshStruct()->getMesh()),
  commT(disc->getComm()) {
  initialAverageEdgeLength = ma::getAverageEdgeLength(mesh);
}

AAdapt::UnifRefSizeField::
~UnifRefSizeField() {
}

void
AAdapt::UnifRefSizeField::computeError() {
}

void
AAdapt::UnifRefSizeField::setParams(double element_size, double err_bound,
				    const std::string& state_var_name) {

  elem_size = element_size;

}

double AAdapt::UnifRefSizeField::getValue(ma::Entity* v) {
  return 0.5 * initialAverageEdgeLength;
}

