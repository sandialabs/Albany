//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_MESHSIZEFIELD_HPP
#define AADAPT_MESHSIZEFIELD_HPP

namespace AAdapt {
/*! \brief Define methods that Albany wants a size field to implement in
 *         addition to those in ma::SizeField.
 */
struct MeshSizeField {
  virtual void setParams(
      const Teuchos::RCP<Teuchos::ParameterList>& p) = 0;
  virtual void computeError() = 0;
  virtual void copyInputFields() = 0;
  virtual void freeInputFields() = 0;
  virtual void freeSizeField() = 0;
};
}

#endif
