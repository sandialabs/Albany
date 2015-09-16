//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_UNIFREFSIZEFIELD_HPP
#define AADAPT_UNIFREFSIZEFIELD_HPP

#include "AAdapt_MeshSizeField.hpp"

namespace AAdapt {

class UnifRefSizeField : public MeshSizeField {

  public:

    UnifRefSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc);

    ~UnifRefSizeField();

    void configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void computeError();

    void copyInputFields();
    void freeInputFields() {}
    void freeSizeField() {}

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

