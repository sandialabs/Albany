//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_MULTIVECTOROUTPUTFILEFACTORY_HPP
#define MOR_MULTIVECTOROUTPUTFILEFACTORY_HPP

#include "MOR_MultiVectorOutputFile.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include <string>

namespace MOR {

class MultiVectorOutputFileFactory {
public:
  explicit MultiVectorOutputFileFactory(const Teuchos::RCP<Teuchos::ParameterList> &params);
  Teuchos::RCP<MultiVectorOutputFile> create();

private:
  Teuchos::RCP<Teuchos::ParameterList> params_;

  std::string outputFileName_, outputFileFormat_;
  Teuchos::Array<std::string> validFileFormats_;

  void initValidFileFormats();
  void initOutputFileFormat();
  void initOutputFileName();

  // Disallow copy and assignment
  MultiVectorOutputFileFactory(const MultiVectorOutputFileFactory &);
  MultiVectorOutputFileFactory &operator=(const MultiVectorOutputFileFactory &);
};

} // namespace MOR

#endif /* MOR_MULTIVECTOROUTPUTFILEFACTORY_HPP */
