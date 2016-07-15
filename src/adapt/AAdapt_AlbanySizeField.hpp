//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_ALBANYSIZEFIELD_HPP
#define AADAPT_ALBANYSIZEFIELD_HPP

#include "AAdapt_MeshAdaptMethod.hpp"

/*
An Albany evaluator calculates the size field and passes it to this class
to perform the adaptation.
*/

namespace AAdapt {

class AlbanySizeField : public MeshAdaptMethod {

  public:

    AlbanySizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc);

    ~AlbanySizeField();

    void adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p){}
    void preProcessShrunkenMesh(){}
    void preProcessOriginalMesh();
    void postProcessFinalMesh();
    void postProcessShrunkenMesh() {}

  private:

    Teuchos::RCP<Albany::APFDiscretization> apfDisc;

};

}

#endif

