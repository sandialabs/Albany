//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_PUMIOUTPUT_HPP
#define ALBANY_PUMIOUTPUT_HPP

#include "Teuchos_RCP.hpp"
#include "Albany_APFMeshStruct.hpp"

namespace Albany {

class PUMIOutput {
  public:
    virtual ~PUMIOutput();
    virtual void writeFile(const double time) = 0;
    virtual void setFileName(const std::string& fname) = 0;
    static PUMIOutput* create(const Teuchos::RCP<APFMeshStruct>& meshStruct,
        const Teuchos::RCP<const Teuchos_Comm>& comm);
};

}

#endif
