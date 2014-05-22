//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_LAYOUTS_HPP
#define AERAS_LAYOUTS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"

#include "Phalanx.hpp"
#include "Albany_DataTypes.hpp"
#include "Albany_Layouts.hpp"

namespace Aeras {
  /*!
   * \brief Struct to construct and hold DataLayouts
   */
  struct Layouts : public Albany::Layouts {
    //Layouts(int worksetSize, int  numVertices, int numNodes, int numQPts, int numDim, int vecDim=-1, int numFace=0);
    Layouts(int worksetSize, int  numVertices, int numNodes, int numQPts, int numDim, int vecDim, int numLevels);
    //! Data Layout for scalar quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_scalar_level;
    //! Data Layout for gradient quantity that lives at quad points
    Teuchos::RCP<PHX::DataLayout> qp_gradient_level;
    //! Data Layout for scalar quantity that lives at node points
    Teuchos::RCP<PHX::DataLayout> node_scalar_level;
  };
}

#endif 
