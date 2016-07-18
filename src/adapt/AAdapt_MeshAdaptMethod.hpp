//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_MESHADAPTMETHOD_HPP
#define AADAPT_MESHADAPTMETHOD_HPP

#include <ma.h>

#include "Albany_APFDiscretization.hpp"
#include "Albany_StateManager.hpp"

namespace AAdapt {

/*! \brief Encapsulates the different uses of MeshAdapt in Albany
 */
struct MeshAdaptMethod {

  MeshAdaptMethod(const Teuchos::RCP<Albany::APFDiscretization>& disc);

  virtual void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p) = 0;

  virtual void preProcessOriginalMesh() = 0;
  virtual void preProcessShrunkenMesh() = 0;
  virtual void adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) = 0;
  virtual void postProcessShrunkenMesh() = 0;
  virtual void postProcessFinalMesh() = 0;

protected:

  Teuchos::RCP<Albany::APFMeshStruct> mesh_struct;
  Teuchos::RCP<const Teuchos_Comm> commT;

  void setCommonMeshAdaptOptions(
      const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_,
      ma::Input *in);

};

}

#endif
