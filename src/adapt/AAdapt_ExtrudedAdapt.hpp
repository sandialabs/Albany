//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_EXTRUDEDADAPT_HPP
#define AADAPT_EXTRUDEDADAPT_HPP

#include "AAdapt_MeshAdaptMethod.hpp"
#include <maExtrude.h>

namespace AAdapt {

class ExtrudedAdapt : public MeshAdaptMethod {
  public:
    ExtrudedAdapt(const Teuchos::RCP<Albany::APFDiscretization>& disc);

    ~ExtrudedAdapt();

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void preProcessOriginalMesh();
    void preProcessShrunkenMesh();
    void adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);
    void postProcessShrunkenMesh();
    void postProcessFinalMesh();

  private:
    MeshAdaptMethod* helper;
    ma::Mesh* mesh;
    ma::ModelExtrusions model_extrusions;
    size_t nlayers;
};

}

#endif
