//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GOALDISCRETIZATION_HPP
#define ALBANY_GOALDISCRETIZATION_HPP

#include "Albany_PUMIDiscretization.hpp"
#include "Albany_GOALMeshStruct.hpp"

namespace Albany {

class GOALDiscretization : public PUMIDiscretization {
  public:

    //! Constructor
    GOALDiscretization(
       Teuchos::RCP<Albany::GOALMeshStruct> goalMeshStruct,
       const Teuchos::RCP<const Teuchos_Comm>& commT,
       const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes = Teuchos::null);

    //! Destructor
    ~GOALDiscretization();

    // Retrieve mesh struct
    Teuchos::RCP<Albany::GOALMeshStruct> getGOALMeshStruct() {return goalMeshStruct;}

  private:

    Teuchos::RCP<Albany::GOALMeshStruct> goalMeshStruct;
};

}

#endif
