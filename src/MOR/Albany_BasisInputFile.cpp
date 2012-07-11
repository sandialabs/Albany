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

#include "Albany_BasisInputFile.hpp"

#include "Albany_MultiVectorInputFile.hpp"
#include "Albany_MultiVectorInputFileFactory.hpp"

#include "Teuchos_Ptr.hpp"

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::Ptr;
using ::Teuchos::nonnull;
using ::Teuchos::ParameterList;

RCP<ParameterList> fillDefaultBasisInputParams(const RCP<ParameterList> &params)
{
  params->get("Input File Group Name", "basis");
  params->get("Input File Default Base File Name", "basis");
  return params;
}

RCP<Epetra_MultiVector> readOrthonormalBasis(const Epetra_Map &basisMap,
                                             const RCP<ParameterList> &fileParams)
{
  MultiVectorInputFileFactory factory(fileParams);
  const RCP<MultiVectorInputFile> file = factory.create();

  const Ptr<const int> maxVecCount(fileParams->getPtr<int>("Basis Size Max"));
  if (nonnull(maxVecCount)) {
    return file->readPartial(basisMap, *maxVecCount);
  } else {
    return file->read(basisMap);
  }
}

} // namespace Albany
