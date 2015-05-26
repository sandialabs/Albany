//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_NONUNIFREFSIZEFIELD_HPP
#define AADAPT_NONUNIFREFSIZEFIELD_HPP

#include "Albany_AbstractPUMIDiscretization.hpp"
#include <ma.h>
#include "Albany_StateManager.hpp"
#include "AAdapt_MeshSizeField.hpp"

namespace AAdapt {

class NonUnifRefSizeField : public ma::IsotropicFunction, public MeshSizeField {

  public:
    NonUnifRefSizeField(const Teuchos::RCP<Albany::AbstractPUMIDiscretization>& disc);

    ~NonUnifRefSizeField();

    ma::Input *configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

    double getValue(ma::Entity* v);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void computeError();

    void copyInputFields();
    void freeInputFields() {}
    void freeSizeField() {}

  private:

};

}

#endif

