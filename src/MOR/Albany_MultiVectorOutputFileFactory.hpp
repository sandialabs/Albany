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

#ifndef ALBANY_MULTIVECTOROUTPUTFILEFACTORY_HPP
#define ALBANY_MULTIVECTOROUTPUTFILEFACTORY_HPP

#include "Albany_MultiVectorOutputFile.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include <string>

namespace Albany {

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

} // end namespace Albany

#endif /* ALBANY_MULTIVECTOROUTPUTFILEFACTORY_HPP */
