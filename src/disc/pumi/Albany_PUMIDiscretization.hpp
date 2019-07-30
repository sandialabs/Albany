//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_PUMIDISCRETIZATION_HPP
#define ALBANY_PUMIDISCRETIZATION_HPP

#include "Albany_APFDiscretization.hpp"
#include "Albany_PUMIMeshStruct.hpp"

namespace Albany {

class PUMIDiscretization : public APFDiscretization {
  public:

    //! Constructor
    PUMIDiscretization(
       Teuchos::RCP<Albany::PUMIMeshStruct> pumiMeshStruct,
       const Teuchos::RCP<const Teuchos_Comm>& commT,
       const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes = Teuchos::null);

    //! Destructor
    ~PUMIDiscretization();

    //! Retrieve mesh struct
    Teuchos::RCP<Albany::PUMIMeshStruct> getPUMIMeshStruct() {return pumiMeshStruct;}

    //! Set restart data
    void setRestartData();

    //! Set data for LandIce problems
    void setLandIceData();

#if defined(ALBANY_LCM)
    WorksetArray<Teuchos::ArrayRCP<double*>>::type const& getBoundaryIndicator() const 
    {//do nothing
    };  

    void printElemGIDws(std::ostream& os) const 
    {//do nothing 
    }; 

    std::map<std::pair<int, int>, GO>
    buildElemWsLIDGIDMap() const 
    {//do nothing
    };

    void
    printWsElNodeID(std::ostream& os) const {}; 
#endif

  private:

    Teuchos::RCP<Albany::PUMIMeshStruct> pumiMeshStruct;
};

}

#endif // ALBANY_PUMIDISCRETIZATION_HPP
