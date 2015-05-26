//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_SPRSIZEFIELD_HPP
#define AADAPT_SPRSIZEFIELD_HPP

#include "Albany_PUMIDiscretization.hpp"
#include <ma.h>
#include "Albany_StateManager.hpp"
#include "AAdapt_MeshSizeField.hpp"

namespace AAdapt {

class SPRSizeField : public ma::IsotropicFunction, public MeshSizeField {

  public:
    SPRSizeField(const Teuchos::RCP<Albany::AbstractPUMIDiscretization>& disc);
  
    ~SPRSizeField();

    ma::Input *configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_)
       { return MeshSizeField::configure(adapt_params_); }

    double getValue(ma::Entity* v);

    int getCubatureDegree(int num_qp);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void computeError();

    void copyInputFields();
    void freeInputFields();
    void freeSizeField();

  private:

    apf::Field* field;
    Albany::StateArrayVec& esa;
    Albany::WsLIDList& elemGIDws;
    Teuchos::RCP<Albany::AbstractPUMIDiscretization> pumi_disc;

    std::string sv_name;
    double rel_err;

    apf::GlobalNumbering* global_numbering;

    int num_qp;
    int cub_degree;

    void getFieldFromStateVariable(apf::Field* eps);
    void computeErrorFromRecoveredGradients();
    void computeErrorFromStateVariable();

};

}

#endif
