//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_SPRSIZEFIELD_HPP
#define AADAPT_SPRSIZEFIELD_HPP

#include "AAdapt_MeshSizeField.hpp"

namespace AAdapt {

class SPRSizeField : public MeshSizeField {

  public:
    SPRSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc);
  
    ~SPRSizeField();

    void configure(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

    int getCubatureDegree(int num_qp);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void computeError();

    void copyInputFields();
    void freeInputFields();
    void freeSizeField();

    class SPRIsoFunc : public ma::IsotropicFunction
    {
      public:
        virtual ~SPRIsoFunc(){}

    /** \brief get the desired element size at this vertex */

        virtual double getValue(ma::Entity* v){

            return apf::getScalar(field,v,0);

        } 

        apf::Field* field;

    } sprIsoFunc;

  private:

    Albany::StateArrayVec& esa;
    Albany::WsLIDList& elemGIDws;
    Teuchos::RCP<Albany::APFDiscretization> pumi_disc;

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
