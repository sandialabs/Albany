//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Intrepid_MiniTensor.h>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM{
  //**********************************************************************
  template<typename EvalT, typename Traits>
  TvergaardHutchinson<EvalT, Traits>::
  TvergaardHutchinson(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
    cubature        (p.get<Teuchos::RCP<Intrepid::Cubature<RealType>> >("Cubature")),
    intrepidBasis   (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType>> >>("Intrepid Basis")),
    jump(p.get<std::string>("Vector Jump Name"),dl->qp_vector),
    currentBasis(p.get<std::string>("Current Basis Name"),dl->qp_tensor),
    cohesiveTraction(p.get<std::string>("Cohesive Traction Name"),dl->qp_vector),
    delta_1(p.get<RealType>("delta_1 Name")),
    delta_2(p.get<RealType>("delta_2 Name")),
    delta_c(p.get<RealType>("delta_c Name")),
    sigma_c(p.get<RealType>("sigma_c Name")),
    beta_0(p.get<RealType>("beta_0 Name")),
    beta_1(p.get<RealType>("beta_1 Name")),
    beta_2(p.get<RealType>("beta_2 Name"))
  {
    this->addDependentField(jump);
    this->addDependentField(currentBasis);
    this->addEvaluatedField(cohesiveTraction);

    this->setName("TvergaardHutchinson" + PHX::typeAsString<EvalT>());

    // get dimension
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_vector->dimensions(dims);
    worksetSize = dims[0];
    numDims = dims[2];

    numQPs = cubature->getNumPoints();

  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void TvergaardHutchinson<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(jump,fm);
    this->utils.setFieldData(currentBasis,fm);
    this->utils.setFieldData(cohesiveTraction,fm);
  }
  //**********************************************************************
  template<typename EvalT, typename Traits>
  void TvergaardHutchinson<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {

    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int pt = 0; pt < numQPs; ++pt) {

        //current basis vector
        Intrepid::Vector<ScalarT> g_0(3, currentBasis,cell, pt,0,0);
        Intrepid::Vector<ScalarT> g_1(3, currentBasis,cell, pt, 1,0);
        Intrepid::Vector<ScalarT> n(3,   currentBasis,cell, pt, 2,0);

        //construct orthogonal unit basis
        Intrepid::Vector<ScalarT> t_0(0,0,0), t_1(0,0,0);
        t_0 = g_0 / norm(g_0);
        t_1 = cross(n,t_0);

        //construct transformation matrix Q (2nd order tensor)
        Intrepid::Tensor<ScalarT> Q(3, Intrepid::ZEROS);
        // manually fill Q = [t_0; t_1; n];
        Q(0,0) = t_0(0); Q(1,0) = t_0(1);  Q(2,0) = t_0(2);
        Q(0,1) = t_1(0); Q(1,1) = t_1(1);  Q(2,1) = t_1(2);
        Q(0,2) = n(0);   Q(1,2) = n(1);    Q(2,2) = n(2);

        //global and local jump
        Intrepid::Vector<ScalarT> jump_global(3, jump,cell, pt, 0);
        Intrepid::Vector<ScalarT> jump_local(3);
        jump_local = Intrepid::dot(Intrepid::transpose(Q), jump_global);

        // matrix beta that controls relative effect of shear and normal opening
        Intrepid::Tensor<ScalarT> beta(3, Intrepid::ZEROS);
        beta(0,0) = beta_0; beta(1,1) = beta_1; beta(2,2) = beta_2;

        // compute scalar effective jump
        ScalarT jump_eff, tmp2;
        Intrepid::Vector<ScalarT> tmp1;
        tmp1 = Intrepid::dot(beta,jump_local);
        tmp2 = Intrepid::dot(jump_local,tmp1);

        jump_eff =
            std::sqrt(Intrepid::dot(jump_local,Intrepid::dot(beta,jump_local)));

        // traction-separation law from Tvergaard-Hutchinson 1992
        ScalarT sigma_eff;
        //Sacado::ScalarValue<ScalarT>::eval
        if(jump_eff <= delta_1)
          sigma_eff = sigma_c * jump_eff / delta_1;
        else if(jump_eff > delta_1 && jump_eff <= delta_2)
          sigma_eff = sigma_c;
        else if(jump_eff > delta_2 && jump_eff <= delta_c)
          sigma_eff = sigma_c * (delta_c - jump_eff) / (delta_c - delta_2);
        else
          sigma_eff = 0.0;

        // construct traction vector
        Intrepid::Vector<ScalarT> traction_local(3);
        traction_local.clear();
        if(jump_eff != 0)
          traction_local = Intrepid::dot(beta,jump_local) * sigma_eff / jump_eff;

        // global traction vector
        Intrepid::Vector<ScalarT> traction_global(3);
        traction_global = Intrepid::dot(Q, traction_local);

        cohesiveTraction(cell,pt,0) = traction_global(0);
        cohesiveTraction(cell,pt,1) = traction_global(1);
        cohesiveTraction(cell,pt,2) = traction_global(2);

        // output for debug, I'll keep it for now until the implementation fully verified
        // Q.Chen
        //std::cout << "jump_eff " << Sacado::ScalarValue<ScalarT>::eval(jump_eff) << std::endl;
        //std::cout << "sigma_eff " << Sacado::ScalarValue<ScalarT>::eval(sigma_eff) << std::endl;

      }
    }
  }
  //**********************************************************************
}
