//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_UNIFREFSIZEFIELD_HPP
#define AADAPT_UNIFREFSIZEFIELD_HPP

#include "AlbPUMI_AbstractPUMIDiscretization.hpp"
#include <ma.h>
#include "Albany_StateManager.hpp"
#include "AAdapt_MeshSizeField.hpp"

namespace AAdapt {

class UnifRefSizeField : public ma::IsotropicFunction, public MeshSizeField {

  public:
    UnifRefSizeField(const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& disc);

    ~UnifRefSizeField();

    double getValue(ma::Entity* v);

    void setParams(double element_size, double err_bound,
		   const std::string& state_var_name);

    void computeError();

    void copyInputFields() {}
    void freeInputFields() {}
    void freeSizeField() {}

  private:

    Teuchos::RCP<const Teuchos_Comm> commT;

    double elem_size;
    double initialAverageEdgeLength;
    apf::Mesh2* mesh;

};

}

#endif

