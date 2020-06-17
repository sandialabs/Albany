//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef INITIAL_CONDITION_HPP
#define INITIAL_CONDITION_HPP

#include "Albany_DataTypes.hpp"
#include "Albany_DiscretizationUtils.hpp"

#include <string>
#include "Teuchos_ParameterList.hpp"

namespace Albany {

void InitialConditions (const Teuchos::RCP<Thyra_Vector>& soln,
                        const Albany::Conn& wsElNodeEqID,
                        const Teuchos::ArrayRCP<std::string>& wsEBNames,
                        const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords,
                        const int neq, const int numDim,
                        Teuchos::ParameterList& icParams,
                        const bool gasRestartSolution = false);

} // namespace Albany

#endif // INITIAL_CONDITION_HPP
