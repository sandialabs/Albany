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
    Layouts(const int worksetSize, 
            const int numVertices, 
            const int numNodes, 
            const int numQPts, 
            const int numDim, 
            const int vecDim, 
            const int numLevels);
    Teuchos::RCP<PHX::DataLayout> qp_scalar_level;
    Teuchos::RCP<PHX::DataLayout> qp_vector_level;
    Teuchos::RCP<PHX::DataLayout> qp_gradient_level;
    Teuchos::RCP<PHX::DataLayout> node_scalar_level;
    Teuchos::RCP<PHX::DataLayout> node_vector_level;
    Teuchos::RCP<PHX::DataLayout> node_qp_tensor;
  };
}

#endif 
