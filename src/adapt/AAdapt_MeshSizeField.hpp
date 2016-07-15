//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_MESHSIZEFIELD_HPP
#define AADAPT_MESHSIZEFIELD_HPP

#include <ma.h>

#include "Albany_APFDiscretization.hpp"
#include "Albany_StateManager.hpp"

namespace AAdapt {
/*! \brief Define methods that Albany wants a size field to implement in
 *         addition to those in ma::SizeField.
 */
struct MeshSizeField {

  MeshSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc);

  virtual void configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) = 0;

  virtual void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p) = 0;
  virtual void computeError() = 0;
  virtual void copyInputFields() = 0;
  virtual void freeInputFields() = 0;
  virtual void freeSizeField() = 0;

protected:

  Teuchos::RCP<Albany::APFMeshStruct> mesh_struct;
  Teuchos::RCP<const Teuchos_Comm> commT;

  void setMAInputParams(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_,
                        ma::Input *in);

};

}

#endif
