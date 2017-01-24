//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_SPRSIZEFIELD_HPP
#define AADAPT_SPRSIZEFIELD_HPP

#include "AAdapt_MeshAdaptMethod.hpp"

namespace AAdapt {

class SPRSizeField : public MeshAdaptMethod {

  public:
    SPRSizeField(const Teuchos::RCP<Albany::APFDiscretization>& disc);

    void adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void preProcessShrunkenMesh();

    void preProcessOriginalMesh();
    void postProcessFinalMesh();
    void postProcessShrunkenMesh();

    class SPRIsoFunc : public ma::IsotropicFunction
    {
      public:
        apf::Field* field;
        virtual ~SPRIsoFunc(){}
    /** \brief get the desired element size at this vertex */
        virtual double getValue(ma::Entity* v){
            return apf::getScalar(field,v,0);
        }
    } sprIsoFunc;

    void setSolName(std::string const& new_sol_name) {
      sol_name = new_sol_name;
    }

  private:

    Albany::WsLIDList& elemGIDws;
    Teuchos::RCP<Albany::APFDiscretization> pumi_disc;

    std::string state_name;
    std::string sol_name;
    bool using_state;
    bool using_rel_err;
    double rel_err;
    size_t target_count;

    int cub_degree;

    void getFieldFromStateVariable(apf::Field* eps);
    void computeErrorFromRecoveredGradients();
    void computeErrorFromStateVariable();
    apf::Field* runSPR(apf::Field* elem_fld);

};

}

#endif
