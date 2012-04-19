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

#include "Albany_ProjectionError.hpp"

#include "Albany_MultiVectorInputFile.hpp"
#include "Albany_MultiVectorInputFileFactory.hpp"

#include "Epetra_Comm.h"
#include "Epetra_LocalMap.h"
#include "Epetra_Vector.h"

#include "Teuchos_Assert.hpp"

// TODO remove dependency
#include <iostream>

namespace Albany {

using Teuchos::RCP;
using Teuchos::ParameterList;

ProjectionError::ProjectionError(const RCP<ParameterList> &params,
                                 const RCP<const Epetra_Map> &dofMap) :
  params_(params),
  dofMap_(dofMap),
  orthonormalBasis_(createOrthonormalBasis())
{
  // Nothing to do
}

RCP<Epetra_MultiVector> ProjectionError::createOrthonormalBasis()
{
  // TODO read partial basis
  MultiVectorInputFileFactory factory(params_); 
  const RCP<MultiVectorInputFile> file = factory.create();
  return file->vectorNew(*dofMap_);
}

void ProjectionError::process(const Epetra_MultiVector &v)
{
  TEUCHOS_ASSERT(!orthonormalBasis_.is_null());
  TEUCHOS_ASSERT_EQUALITY(v.GlobalLength(), orthonormalBasis_->GlobalLength());

  const int ZERO_BASED_INDEXING = 0;
  const bool NO_INIT = false;
 
  Epetra_LocalMap componentMap(orthonormalBasis_->NumVectors(), ZERO_BASED_INDEXING, dofMap_->Comm());
  Epetra_MultiVector components(componentMap, v.NumVectors(), NO_INIT);

  // components <- orthonormalBasis^T * v
  components.Multiply('T', 'N', 1.0, *orthonormalBasis_, v, 0.0);

  // absoluteError <- v - orthonormalBasis * components
  Epetra_MultiVector absoluteError = v;
  absoluteError.Multiply('N', 'N', -1.0, *orthonormalBasis_, components, 1.0);

  // Norm computations
  Epetra_LocalMap normMap(components.NumVectors(), ZERO_BASED_INDEXING, dofMap_->Comm());

  Epetra_Vector absoluteErrorNorm(normMap, NO_INIT);
  absoluteError.Norm2(absoluteErrorNorm.Values());

  Epetra_Vector referenceNorm(normMap, NO_INIT);
  v.Norm2(referenceNorm.Values());

  // ||relative_error|| <- ||absolute_error|| / ||v||
  Epetra_Vector relativeErrorNorm(normMap, NO_INIT);
  relativeErrorNorm.ReciprocalMultiply(1.0, referenceNorm, absoluteErrorNorm, 0.0);

  // Write to standard output
  // TODO file output
  components.Print(std::cout);
  referenceNorm.Print(std::cout);
  absoluteErrorNorm.Print(std::cout);
  relativeErrorNorm.Print(std::cout);
}

} // end namespace Albany
