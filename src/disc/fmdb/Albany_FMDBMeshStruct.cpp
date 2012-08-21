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

#include <iostream>

#include "Albany_FMDBMeshStruct.hpp"
#include "Teuchos_VerboseObject.hpp"


#include <Shards_BasicTopologies.hpp>



Albany::FMDBMeshStruct::FMDBMeshStruct(
          const Teuchos::RCP<Teuchos::ParameterList>& params,
		  const Teuchos::RCP<const Epetra_Comm>& comm) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{

  params->validateParameters(*getValidDiscretizationParameters(),0);

}


void
Albany::FMDBMeshStruct::setFieldAndBulkData(
                  const Teuchos::RCP<const Epetra_Comm>& comm,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize)
{

}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::FMDBMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL;
/*
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getValidGenericSTKParameters("Valid FMDBParams");
  validPL->set<string>("FMDB Input File Name", "", "File Name For FMDB Mesh Input");
  validPL->set<string>("FMDB Output File Name", "", "File Name For FMDB Mesh Output");
  validPL->set<int>("Restart Index", 1, "FMDB time index to read for inital guess/condition.");
*/

  return validPL;
}
