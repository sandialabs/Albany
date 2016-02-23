//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_UNIFSIZEFIELD_HPP
#define AADAPT_UNIFSIZEFIELD_HPP

#include "AAdapt_MeshSizeField.hpp"

namespace AAdapt {

class UnifSizeField : public MeshSizeField {

  public:

    UnifSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc);

    ~UnifSizeField();

    void configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void computeError();

    void copyInputFields() {}
    void freeInputFields() {}
    void freeSizeField() {}

    class UnifIsoFunc : public ma::IsotropicFunction
    {
      public:
        virtual ~UnifIsoFunc(){}

    /** \brief get the desired element size at this vertex */

        virtual double getValue(ma::Entity* vert){
           return elem_size;
        } 

        double elem_size;

    } unifIsoFunc;

};

}

#endif

