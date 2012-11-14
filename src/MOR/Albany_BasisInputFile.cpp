//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
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
