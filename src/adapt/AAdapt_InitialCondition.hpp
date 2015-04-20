//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_INITIALCONDITION_HPP
#define AADAPT_INITIALCONDITION_HPP

#include <string>
#include "Teuchos_ParameterList.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Vector.h"
#endif
#include "Albany_DataTypes.hpp"

namespace AAdapt {

#if defined(ALBANY_EPETRA)
void InitialConditions(const Teuchos::RCP<Epetra_Vector>& soln,
                       const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >& wsElNodeEqID,
                       const Teuchos::ArrayRCP<std::string>& wsEBNames,
                       const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords,
                       const int neq, const int numDim,
                       Teuchos::ParameterList& icParams,
                       const bool gasRestartSolution = false);
#endif

void InitialConditionsT(const Teuchos::RCP<Tpetra_Vector>& solnT,
                       const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >& wsElNodeEqID,
                       const Teuchos::ArrayRCP<std::string>& wsEBNames,
                       const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords,
                       const int neq, const int numDim,
                       Teuchos::ParameterList& icParams,
                       const bool gasRestartSolution = false);

}
#endif
