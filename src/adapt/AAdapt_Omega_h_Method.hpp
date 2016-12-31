//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AADAPT_OMEGA_H_METHOD_HPP
#define AADAPT_OMEGA_H_METHOD_HPP

#include "AAdapt_MeshAdaptMethod.hpp"

#include <Omega_h.hpp>

namespace AAdapt {

class Omega_h_Method : public MeshAdaptMethod {
  public:
    Omega_h_Method(const Teuchos::RCP<Albany::APFDiscretization>& disc);

    ~Omega_h_Method();

    void setParams(const Teuchos::RCP<Teuchos::ParameterList>& p);

    void preProcessOriginalMesh();
    void preProcessShrunkenMesh();
    void adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_);
    void postProcessShrunkenMesh();
    void postProcessFinalMesh();

  private:
    std::string size_method;
    double maximum_size;
    bool should_target_count;
    double target_count;
    bool should_target_error;
    double target_error;
    bool should_limit_gradation;
    double gradation_rate;
    bool should_prevent_overshoot;
    double overshoot_allowance;
    MeshAdaptMethod* helper;
    ma::Mesh* mesh_apf;
    Omega_h::Library library_osh;
    Omega_h::Mesh mesh_osh;
};

}

#endif
