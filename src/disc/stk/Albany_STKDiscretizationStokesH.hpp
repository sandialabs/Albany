//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_STKDISCRETIZATIONSTOKESH_HPP
#define ALBANY_STKDISCRETIZATIONSTOKESH_HPP

#include <vector>
#include <utility>


#include "Albany_STKDiscretization.hpp"



namespace Albany {

  class STKDiscretizationStokesH : public Albany::STKDiscretization {
  public:

    //! Constructor
    STKDiscretizationStokesH(
       Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct,
       const Teuchos::RCP<const Teuchos_Comm>& commT,
       const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes = Teuchos::null);


    //! Destructor
    ~STKDiscretizationStokesH();

  private:
    //! Process STK mesh for CRS Graphs
    void computeGraphs();
  };

}

#endif // ALBANY_STKDISCRETIZATION_HPP
