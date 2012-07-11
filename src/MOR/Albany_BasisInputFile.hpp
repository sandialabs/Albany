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

#ifndef ALBANY_BASISINPUTFILE_HPP
#define ALBANY_BASISINPUTFILE_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

class Epetra_MultiVector;
class Epetra_Map;

namespace Albany {

Teuchos::RCP<Teuchos::ParameterList> fillDefaultBasisInputParams(
    const Teuchos::RCP<Teuchos::ParameterList> &params);

Teuchos::RCP<Epetra_MultiVector> readOrthonormalBasis(
    const Epetra_Map &basisMap,
    const Teuchos::RCP<Teuchos::ParameterList> &fileParams);

} // namespace Albany

#endif /* ALBANY_BASISINPUTFILE_HPP */
