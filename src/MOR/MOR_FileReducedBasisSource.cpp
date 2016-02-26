//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MOR_FileReducedBasisSource.hpp"

#include "MOR_MultiVectorInputFile.hpp"
#include "MOR_MultiVectorInputFileFactory.hpp"
#include "MOR_InputFileEpetraMVSource.hpp"

#include "Epetra_MultiVector.h"

#include "Teuchos_Ptr.hpp"

namespace MOR {

namespace Detail {

Teuchos::RCP<Teuchos::ParameterList>
fillDefaultBasisInputParams(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  const std::string defaultFileName = "basis";
  params->get("Input File Group Name", defaultFileName);
  params->get("Input File Default Base File Name", defaultFileName);
  return params;
}

} // end namespace Detail


EpetraMVSourceInputFileProvider::EpetraMVSourceInputFileProvider(const Epetra_Map &vectorMap) :
  vectorMap_(vectorMap)
{}

Teuchos::RCP<BasicEpetraMVSource>
EpetraMVSourceInputFileProvider::operator()(const Teuchos::RCP<Teuchos::ParameterList> &params)
{
  MultiVectorInputFileFactory factory(Detail::fillDefaultBasisInputParams(params));
  const Teuchos::RCP<MultiVectorInputFile> file = factory.create();
  return Teuchos::rcp(new InputFileEpetraMVSource(vectorMap_, file));
}


FileReducedBasisSource::FileReducedBasisSource(const Epetra_Map &basisMap) :
  TruncatedReducedBasisSource<EpetraMVSourceInputFileProvider>(basisMap)
{}

} // namespace MOR
