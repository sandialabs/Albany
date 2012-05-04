/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#ifndef ALBANY_REDUCEDORDERMODELFACTORY_HPP
#define ALBANY_REDUCEDORDERMODELFACTORY_HPP

#include "EpetraExt_ModelEvaluator.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

class Epetra_MultiVector;
class Epetra_Map;

namespace Albany {

class ReducedOrderModelFactory {
public:
  explicit ReducedOrderModelFactory(const Teuchos::RCP<Teuchos::ParameterList> &parentParams);

  Teuchos::RCP<EpetraExt::ModelEvaluator> create(const Teuchos::RCP<EpetraExt::ModelEvaluator> &child);

private:
  Teuchos::RCP<Teuchos::ParameterList> params_;

  static Teuchos::RCP<Teuchos::ParameterList> extractModelOrderReductionParams(const Teuchos::RCP<Teuchos::ParameterList> &source);
  static Teuchos::RCP<Teuchos::ParameterList> extractReducedOrderModelParams(const Teuchos::RCP<Teuchos::ParameterList> &source);
  static Teuchos::RCP<Teuchos::ParameterList> fillDefaultReducedOrderModelParams(const Teuchos::RCP<Teuchos::ParameterList> &source);

  bool useReducedOrderModel() const;
  
  static Teuchos::RCP<Epetra_MultiVector> createOrthonormalBasis(const Epetra_Map &fullStateMap,
                                                                 const Teuchos::RCP<Teuchos::ParameterList> &params);

  // Disallow copy & assignment
  ReducedOrderModelFactory(const ReducedOrderModelFactory &);
  ReducedOrderModelFactory &operator=(const ReducedOrderModelFactory &);
};

} // end namespace Albany

#endif /* ALBANY_REDUCEDORDERMODELFACTORY_HPP */
