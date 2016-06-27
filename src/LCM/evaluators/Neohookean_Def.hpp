//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid2_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include <typeinfo>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  Neohookean<EvalT, Traits>::
  Neohookean(const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) :
    defGrad          (p.get<std::string>("DefGrad Name"), dl->qp_tensor),
    J                (p.get<std::string>("DetDefGrad Name"), dl->qp_scalar),
    elasticModulus   (p.get<std::string>("Elastic Modulus Name"), dl->qp_scalar),
    poissonsRatio    (p.get<std::string>("Poissons Ratio Name"), dl->qp_scalar),
    stress           (p.get<std::string>("Stress Name"), dl->qp_tensor)
  {
    // Pull out numQPs and numDims from a Layout
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_tensor->dimensions(dims);
    numQPs  = dims[1];
    numDims = dims[2];
    worksetSize = dims[0];

    this->addDependentField(defGrad);
    this->addDependentField(J);
    this->addDependentField(elasticModulus);
    this->addDependentField(poissonsRatio);

    this->addEvaluatedField(stress);

    this->setName("Neohookean Stress"+PHX::typeAsString<EvalT>());

    // initilize Tensors
    F = Intrepid2::Tensor<ScalarT>(numDims);
    b = Intrepid2::Tensor<ScalarT>(numDims);
    sigma = Intrepid2::Tensor<ScalarT>(numDims);
    I = Intrepid2::eye<ScalarT>(numDims);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Neohookean<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(stress,fm);
    this->utils.setFieldData(defGrad,fm);
    this->utils.setFieldData(J,fm);
    this->utils.setFieldData(elasticModulus,fm);
    this->utils.setFieldData(poissonsRatio,fm);
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void Neohookean<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    //bool print = false;
    //if (typeid(ScalarT) == typeid(RealType)) print = true;
    //cout.precision(15);

    ScalarT kappa;
    ScalarT mu;
    ScalarT Jm53;

    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
        kappa = 
          elasticModulus(cell,qp) / ( 3. * ( 1. - 2. * poissonsRatio(cell,qp) ) );
        mu = 
          elasticModulus(cell,qp) / ( 2. * ( 1. + poissonsRatio(cell,qp) ) );

//          TEUCHOS_TEST_FOR_EXCEPTION(J(cell,qp) <= 0, std::runtime_error,
//              " negative / zero volume detected in Neohookean_Def.hpp line " + __LINE__);
// Note - J(cell, qp) < equal to zero causes an FPE (GAH)

        Jm53 = std::pow(J(cell,qp), -5./3.);

        F.fill(defGrad,cell,qp,0,0);
        b = F*transpose(F);
        sigma = 0.5 * kappa * ( J(cell,qp) - 1. / J(cell,qp) ) * I
          + mu * Jm53 * Intrepid2::dev(b);

        for (int i=0; i < numDims; ++i)
          for (int j=0; j < numDims; ++j)
            stress(cell,qp,i,j) = sigma(i,j);
      }
    }
  }
  //----------------------------------------------------------------------------
}

