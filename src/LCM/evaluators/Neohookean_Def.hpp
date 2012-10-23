//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "LCM/utils/Tensor.h"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
Neohookean<EvalT, Traits>::
Neohookean(const Teuchos::ParameterList& p) :
  F                (p.get<std::string>                   ("DefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  J                (p.get<std::string>                   ("DetDefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  elasticModulus   (p.get<std::string>                   ("Elastic Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  poissonsRatio    (p.get<std::string>                   ("Poissons Ratio Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  worksetSize = dims[0];

  this->addDependentField(F);
  this->addDependentField(J);
  this->addDependentField(elasticModulus);
  // PoissonRatio not used in 1D stress calc
  if (numDims>1) this->addDependentField(poissonsRatio);

  this->addEvaluatedField(stress);

  // scratch space FCs
  FT.resize(worksetSize, numQPs, numDims, numDims);

  this->setName("NeoHookean Stress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void Neohookean<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(F,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(elasticModulus,fm);
  if (numDims>1) this->utils.setFieldData(poissonsRatio,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Neohookean<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  cout.precision(15);
  ScalarT kappa;
  ScalarT mu;
  ScalarT Jm53;
  ScalarT trace;
  
  //Intrepid::FieldContainer<ScalarT> b(workset.numCells, numQPs, numDims, numDims);
  //Replaced above line with the following.  The above will cause code to crash in some builds. IK, 4/4/2012
  Intrepid::FieldContainer<ScalarT> b(worksetSize, numQPs, numDims, numDims);
  Intrepid::RealSpaceTools<ScalarT>::transpose(FT, F);
  Intrepid::FunctionSpaceTools::tensorMultiplyDataData<ScalarT> (b, F, FT, 'N');

  switch (numDims) {
  case 1:
    Intrepid::FunctionSpaceTools::tensorMultiplyDataData<ScalarT>(stress, elasticModulus, b);
    break;
  case 2:
  case 3:
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        kappa = elasticModulus(cell,qp) / ( 3. * ( 1. - 2. * poissonsRatio(cell,qp) ) );
        mu    = elasticModulus(cell,qp) / ( 2. * ( 1. + poissonsRatio(cell,qp) ) );
        Jm53  = std::pow(J(cell,qp), -5./3.);
        trace = 0.0;
        for (std::size_t i=0; i < numDims; ++i) trace += (1./numDims) * b(cell,qp,i,i);
        for (std::size_t i=0; i < numDims; ++i) {
          for (std::size_t j=0; j < numDims; ++j) {
            stress(cell,qp,i,j) = mu * Jm53 * ( b(cell,qp,i,j) );
          }
          stress(cell,qp,i,i) += 0.5 * kappa * ( J(cell,qp) - 1. / J(cell,qp) ) - mu * Jm53 * trace;
        }
      }
    }
    break;
  }

  // LCM::Tensor<ScalarT, 3> I( LCM::eye<ScalarT, 3>() );
  
  // for (std::size_t cell=0; cell < workset.numCells; ++cell) {
  //   for (std::size_t qp=0; qp < numQPs; ++qp) {

  //     kappa = elasticModulus(cell,qp) / ( 3. * ( 1. - 2. * poissonsRatio(cell,qp) ) );
  //     mu    = elasticModulus(cell,qp) / ( 2. * ( 1. + poissonsRatio(cell,qp) ) );
      
  //     LCM::Tensor<ScalarT, 3> defGrad( &F(cell,qp,0,0) );
  //     //LCM::Tensor<ScalarT, 3> V, R1, U, R2;
  //     //boost::tie(V,R1) = LCM::polar_left( defGrad );
  //     //boost::tie(R2,U) = LCM::polar_right( defGrad );

  //     LCM::Tensor<ScalarT, 3> R, U, V;
  //     boost::tie(R,U) = LCM::polar(defGrad);
  //     V = R * U * transpose(R);

  //     ScalarT detJ = LCM::det(V);

  //     ScalarT Jm23 = std::pow( detJ, -2.0/3.0 );

  //     ScalarT pressure = 0.5 * kappa * ( detJ - 1.0 / detJ );

  //     LCM::Tensor<ScalarT, 3> b( Jm23 * V * V );

  //     //b *= Jm23;
  //     b -= LCM::trace(b)/3.0*I;

  //     LCM::Tensor<ScalarT, 3> sigma( pressure*I + (mu/detJ) * b );

  //     LCM::Tensor<ScalarT, 3> T( LCM::transpose(R) * sigma * R );

  //     for (std::size_t i=0; i < numDims; ++i) 
  //       for (std::size_t j=0; j < numDims; ++j) 
  //         stress(cell,qp,i,j) = sigma(i,j);
      
      
      // std::cout << "Left Polar\n";

      // std::cout << "V(0,0) " << V(0,0) << endl;
      // std::cout << "V(1,0) " << V(1,0) << endl;
      // std::cout << "V(2,0) " << V(2,0) << endl;
      // std::cout << "V(0,1) " << V(0,1) << endl;
      // std::cout << "V(1,1) " << V(1,1) << endl;
      // std::cout << "V(2,1) " << V(2,1) << endl;
      // std::cout << "V(0,2) " << V(0,2) << endl;
      // std::cout << "V(1,2) " << V(1,2) << endl;
      // std::cout << "V(2,2) " << V(2,2) << endl;
      // std::cout << "R(0,0) " << R1(0,0) << endl;
      // std::cout << "R(1,0) " << R1(1,0) << endl;
      // std::cout << "R(2,0) " << R1(2,0) << endl;
      // std::cout << "R(0,1) " << R1(0,1) << endl;
      // std::cout << "R(1,1) " << R1(1,1) << endl;
      // std::cout << "R(2,1) " << R1(2,1) << endl;
      // std::cout << "R(0,2) " << R1(0,2) << endl;
      // std::cout << "R(1,2) " << R1(1,2) << endl;
      // std::cout << "R(2,2) " << R1(2,2) << endl;

      // //**********************************************************************

      // std::cout << "Right Polar\n";

      // std::cout << "U(0,0) " << U(0,0) << endl;
      // std::cout << "U(1,0) " << U(1,0) << endl;
      // std::cout << "U(2,0) " << U(2,0) << endl;
      // std::cout << "U(0,1) " << U(0,1) << endl;
      // std::cout << "U(1,1) " << U(1,1) << endl;
      // std::cout << "U(2,1) " << U(2,1) << endl;
      // std::cout << "U(0,2) " << U(0,2) << endl;
      // std::cout << "U(1,2) " << U(1,2) << endl;
      // std::cout << "U(2,2) " << U(2,2) << endl;
      // std::cout << "R(0,0) " << R2(0,0) << endl;
      // std::cout << "R(1,0) " << R2(1,0) << endl;
      // std::cout << "R(2,0) " << R2(2,0) << endl;
      // std::cout << "R(0,1) " << R2(0,1) << endl;
      // std::cout << "R(1,1) " << R2(1,1) << endl;
      // std::cout << "R(2,1) " << R2(2,1) << endl;
      // std::cout << "R(0,2) " << R2(0,2) << endl;
      // std::cout << "R(1,2) " << R2(1,2) << endl;
      // std::cout << "R(2,2) " << R2(2,2) << endl;

      // std::cout << "F\n";
      // std::cout << "F(0,0) " << F(cell,qp,0,0) << endl;
      // std::cout << "F(1,1) " << F(cell,qp,1,1) << endl;
      // std::cout << "F(2,2) " << F(cell,qp,2,2) << endl;
      // std::cout << "F(0,1) " << F(cell,qp,0,1) << endl;
      // std::cout << "F(1,2) " << F(cell,qp,1,2) << endl;
      // std::cout << "F(0,2) " << F(cell,qp,0,2) << endl;
      // std::cout << "F(1,0) " << F(cell,qp,1,0) << endl;
      // std::cout << "F(2,1) " << F(cell,qp,2,1) << endl;
      // std::cout << "F(2,0) " << F(cell,qp,2,0) << endl;

      // std::cout << "Stress\n";
      // std::cout << "stress(0,0) " << stress(cell,qp,0,0) << endl;
      // std::cout << "stress(1,1) " << stress(cell,qp,1,1) << endl;
      // std::cout << "stress(2,2) " << stress(cell,qp,2,2) << endl;
      // std::cout << "stress(0,1) " << stress(cell,qp,0,1) << endl;
      // std::cout << "stress(1,2) " << stress(cell,qp,1,2) << endl;
      // std::cout << "stress(0,2) " << stress(cell,qp,0,2) << endl;
      // std::cout << "stress(1,0) " << stress(cell,qp,1,0) << endl;
      // std::cout << "stress(2,1) " << stress(cell,qp,2,1) << endl;
      // std::cout << "stress(2,0) " << stress(cell,qp,2,0) << endl;

  //   }
  // }
}

//**********************************************************************
}

