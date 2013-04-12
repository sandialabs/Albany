//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_BasisInputFile.hpp"

#include "MOR_MultiVectorInputFile.hpp"
#include "MOR_MultiVectorInputFileFactory.hpp"

#include "Teuchos_Ptr.hpp"

namespace MOR {

namespace { // anonymous

Teuchos::RCP<Teuchos::ParameterList>
fillDefaultBasisInputParams(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string defaultFileName = "basis";
  params->get("Input File Group Name", defaultFileName);
  params->get("Input File Default Base File Name", defaultFileName);
  return params;
}

Teuchos::RCP<Epetra_MultiVector>
readOrthonormalBasis(
    const Epetra_Map &basisMap, const Teuchos::RCP<Teuchos::ParameterList> &fileParams)
{
  MultiVectorInputFileFactory factory(fileParams);
  const Teuchos::RCP<MultiVectorInputFile> file = factory.create();

  const Teuchos::Ptr<const int> maxVecCount(fileParams->getPtr<int>("Basis Size Max"));
  if (Teuchos::nonnull(maxVecCount)) {
    return file->readPartial(basisMap, *maxVecCount);
  } else {
    return file->read(basisMap);
  }
}

} // end anonymous namespace

BasisInputFile::BasisInputFile(const Epetra_Map &basisMap) :
  basisMap_(basisMap)
{}

Teuchos::RCP<Epetra_MultiVector>
BasisInputFile::operator()(const Teuchos::RCP<Teuchos::ParameterList> &params) {
  return readOrthonormalBasis(basisMap_, fillDefaultBasisInputParams(params));
}

} // namespace MOR
