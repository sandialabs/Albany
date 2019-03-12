//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_INITIAL_CONDITION_HPP
#define AADAPT_INITIAL_CONDITION_HPP

#include "Albany_DataTypes.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include <string>
#include "Teuchos_ParameterList.hpp"

namespace AAdapt {

void InitialConditions (const Teuchos::RCP<Thyra_Vector>& solnT,
                       const Albany::AbstractDiscretization::Conn& wsElNodeEqID,
                       const Teuchos::ArrayRCP<std::string>& wsEBNames,
                       const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords,
                       const int neq, const int numDim,
                       Teuchos::ParameterList& icParams,
                       const bool gasRestartSolution = false);

} // namespace AAdapt

#endif // AADAPT_INITIAL_CONDITION_HPP
