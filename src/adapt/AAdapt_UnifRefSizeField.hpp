//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_UNIFREFSIZEFIELD_HPP
#define AADAPT_UNIFREFSIZEFIELD_HPP

#include "Albany_AbstractPUMIDiscretization.hpp"
#include <ma.h>
#include "Albany_StateManager.hpp"
#include "AAdapt_MeshSizeField.hpp"

namespace AAdapt {

class UnifRefSizeField : public ma::IsotropicFunction, public MeshSizeField {

  public:
    UnifRefSizeField(const Teuchos::RCP<Albany::AbstractPUMIDiscretization>& disc);

    ~UnifRefSizeField();

    double getValue(ma::Entity* v);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void computeError();

    void copyInputFields() {}
    void freeInputFields() {}
    void freeSizeField() {}

  private:

    Teuchos::RCP<const Teuchos_Comm> commT;

    double elem_size;
    double initialAverageEdgeLength;
    Teuchos::RCP<Albany::PUMIMeshStruct> mesh_struct;

};

}

#endif

