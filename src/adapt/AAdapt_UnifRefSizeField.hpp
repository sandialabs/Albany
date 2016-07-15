//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_UNIFREFSIZEFIELD_HPP
#define AADAPT_UNIFREFSIZEFIELD_HPP

#include "AAdapt_MeshAdaptMethod.hpp"

namespace AAdapt {

class UnifRefSizeField : public MeshAdaptMethod {

  public:

    UnifRefSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc);

    ~UnifRefSizeField();

    void adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void preProcessShrunkenMesh();

    void preProcessOriginalMesh();
    void postProcessFinalMesh() {}
    void postProcessShrunkenMesh() {}

    class UnifRefIsoFunc : public ma::IsotropicFunction
    {
      public:
        virtual ~UnifRefIsoFunc(){}

    /** \brief get the desired element size at this vertex */

        virtual double getValue(ma::Entity* vert){
            return elem_size * averageEdgeLength;
        }

        double elem_size;
        double averageEdgeLength;

    } unifRefIsoFunc;


};

}

#endif

