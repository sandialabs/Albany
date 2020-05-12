//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STK_DISCRETIZATION_STOKES_H_HPP
#define ALBANY_STK_DISCRETIZATION_STOKES_H_HPP

#include "Albany_STKDiscretization.hpp"

namespace Albany {

class STKDiscretizationStokesH : public STKDiscretization {
public:

  //! Constructor
  STKDiscretizationStokesH(
     const Teuchos::RCP<Teuchos::ParameterList>& discParams,
     Teuchos::RCP<AbstractSTKMeshStruct>& stkMeshStruct,
     const Teuchos::RCP<const Teuchos_Comm>& commT,
     const Teuchos::RCP<RigidBodyModes>& rigidBodyModes = Teuchos::null);


  //! Destructor
  ~STKDiscretizationStokesH() = default;

private:
  //! Process STK mesh for CRS Graphs
  void computeGraphs();
};

} // namespace Albany

#endif // ALBANY_STK_DISCRETIZATION_STOKESH_H_HPP
