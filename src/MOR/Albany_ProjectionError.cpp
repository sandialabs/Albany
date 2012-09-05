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

#include "Albany_BasisInputFile.hpp"
#include "Albany_MultiVectorOutputFile.hpp"
#include "Albany_MultiVectorOutputFileFactory.hpp"

#include "Albany_ReducedSpace.hpp"

#include "Epetra_Comm.h"
#include "Epetra_LocalMap.h"
#include "Epetra_Vector.h"

#include "Teuchos_Assert.hpp"

#include <algorithm>
#include <iterator>

// TODO remove dependency
#include <iostream>

namespace Albany {

using ::Teuchos::RCP;
using ::Teuchos::rcp;
using ::Teuchos::ParameterList;

// TODO: Better hiding / encapsulation
const int ZERO_BASED_INDEXING = 0;
const bool NO_INIT = false;

ProjectionError::ProjectionError(const RCP<ParameterList> &params,
                                 const RCP<const Epetra_Map> &dofMap) :
  params_(fillDefaultParams(params)),
  dofMap_(dofMap),
  reducedSpace_()
{
  const RCP<const Epetra_MultiVector> basis = readOrthonormalBasis(*dofMap_, params_);
  reducedSpace_ = rcp(new LinearReducedSpace(*basis));
}

RCP<ParameterList> ProjectionError::fillDefaultParams(const RCP<ParameterList> &params)
{
  fillDefaultBasisInputParams(params);
  params->get("Output File Group Name", "proj_error");
  params->get("Output File Default Base File Name", "proj_error");
  return params;
}

// TODO: Do no actual work in the destructor
ProjectionError::~ProjectionError()
{
#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int GlobalIndex;
#else
  typedef long long GlobalIndex;
#endif
  Epetra_LocalMap entryMap(static_cast<GlobalIndex>(relativeErrorNorms_.size()), ZERO_BASED_INDEXING, dofMap_->Comm());
  Epetra_Vector entries(entryMap, NO_INIT);

  for (int i = 0; i < entries.MyLength(); ++i) {
    entries[i] = relativeErrorNorms_[i];
  }

  MultiVectorOutputFileFactory factory(params_);
  const RCP<MultiVectorOutputFile> file = factory.create();
  file->write(entries);
}

void ProjectionError::process(const Epetra_MultiVector &v)
{
  // components <- orthonormalBasis^T * v
  const RCP<const Epetra_MultiVector> components = reducedSpace_->reduction(v);

  // absoluteError <- orthonormalBasis * components - v
  const RCP<Epetra_MultiVector> absoluteError = reducedSpace_->expansion(*components);
  absoluteError->Update(-1.0, v, 1.0);

  // Norm computations
  Epetra_LocalMap normMap(components->NumVectors(), ZERO_BASED_INDEXING, dofMap_->Comm());

  Epetra_Vector absoluteErrorNorm(normMap, NO_INIT);
  absoluteError->Norm2(absoluteErrorNorm.Values());

  Epetra_Vector referenceNorm(normMap, NO_INIT);
  v.Norm2(referenceNorm.Values());

  // ||relative_error|| <- ||absolute_error|| / ||v||
  Epetra_Vector relativeErrorNorm(normMap, NO_INIT);
  relativeErrorNorm.ReciprocalMultiply(1.0, referenceNorm, absoluteErrorNorm, 0.0);

  // Collect output data
  for (int i = 0; i < relativeErrorNorm.MyLength(); ++i) {
    relativeErrorNorms_.push_back(relativeErrorNorm[i]);
  }

  // Write to standard output
  // TODO remove
  components->Print(std::cout);
  referenceNorm.Print(std::cout);
  absoluteErrorNorm.Print(std::cout);
  relativeErrorNorm.Print(std::cout);
}

} // end namespace Albany
