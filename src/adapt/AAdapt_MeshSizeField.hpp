//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_MESHSIZEFIELD_HPP
#define AADAPT_MESHSIZEFIELD_HPP

#include <ma.h>

#include "Albany_PUMIDiscretization.hpp"
#include "Albany_StateManager.hpp"

namespace AAdapt {
/*! \brief Define methods that Albany wants a size field to implement in
 *         addition to those in ma::SizeField.
 */
struct MeshSizeField {

  MeshSizeField(const Teuchos::RCP<Albany::AbstractPUMIDiscretization>& disc);

  virtual ma::Input* configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

  virtual void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p) = 0;
  virtual void computeError() = 0;
  virtual void copyInputFields() = 0;
  virtual void freeInputFields() = 0;
  virtual void freeSizeField() = 0;

protected:

  Teuchos::RCP<Albany::PUMIMeshStruct> mesh_struct;
  Teuchos::RCP<const Teuchos_Comm> commT;

};

}

#endif
