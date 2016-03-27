//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOR_SINGULARVALUESHELPERS_HPP
#define MOR_SINGULARVALUESHELPERS_HPP

#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayView.hpp"

namespace MOR {

Teuchos::Array<double> computeDiscardedEnergyFractions(Teuchos::ArrayView<const double> singularValues);

} // namespace MOR

#endif /* MOR_SINGULARVALUESHELPERS_HPP */
