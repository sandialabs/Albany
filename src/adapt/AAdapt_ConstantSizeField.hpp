//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_CONSTANTSIZEFIELD_HPP
#define AADAPT_CONSTANTSIZEFIELD_HPP

#include "AAdapt_MeshAdaptMethod.hpp"

namespace AAdapt {

class ConstantSizeField : public MeshAdaptMethod {

  public:

    ConstantSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc);

    ~ConstantSizeField();

    void adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void preProcessShrunkenMesh();

    void preProcessOriginalMesh() {}
    void postProcessFinalMesh() {}
    void postProcessShrunkenMesh() {}

    class ConstantIsoFunc : public ma::IsotropicFunction
    {
      public:
        virtual ~ConstantIsoFunc(){}

    /** \brief get the desired element size at this vertex */

        virtual double getValue(ma::Entity* vert){
           return value_;
        } 

        double value_;

    } constantIsoFunc;

};

}

#endif

