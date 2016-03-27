//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef MOR_MULTIVECTORINPUTFILEFACTORY_HPP
#define MOR_MULTIVECTORINPUTFILEFACTORY_HPP

#include "MOR_MultiVectorInputFile.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include <string>

namespace MOR {

class MultiVectorInputFileFactory {
public:
  explicit MultiVectorInputFileFactory(const Teuchos::RCP<Teuchos::ParameterList> &params);
  Teuchos::RCP<MultiVectorInputFile> create();

private:
  Teuchos::RCP<Teuchos::ParameterList> params_;

  std::string inputFileName_, inputFileFormat_;
  Teuchos::Array<std::string> validFileFormats_;

  void initValidFileFormats();
  void initInputFileFormat();
  void initInputFileName();

  // Disallow copy and assignment
  MultiVectorInputFileFactory(const MultiVectorInputFileFactory &);
  MultiVectorInputFileFactory &operator=(const MultiVectorInputFileFactory &);
};

} // namespace MOR

#endif /* MOR_MULTIVECTORINPUTFILEFACTORY_HPP */
