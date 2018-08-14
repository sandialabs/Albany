//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
//
// TODO:
// 1.  Implement 1D and 2D.

#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace HMC {

//**********************************************************************
template <typename EvalT, typename Traits>
Stresses<EvalT, Traits>::Stresses(const Teuchos::ParameterList& p)
    : strain(
          p.get<std::string>("Strain Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP 2Tensor Data Layout")),
      stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP 2Tensor Data Layout")),
      numMicroScales(p.get<int>("Additional Scales")),
      C11(p.get<RealType>("C11")),
      C33(p.get<RealType>("C33")),
      C12(p.get<RealType>("C12")),
      C23(p.get<RealType>("C23")),
      C44(p.get<RealType>("C44")),
      C66(p.get<RealType>("C66"))
{
  strainDifference.resize(numMicroScales);
  microStrainGradient.resize(numMicroScales);

  microStress.resize(numMicroScales);
  doubleStress.resize(numMicroScales);

  for (int i = 0; i < numMicroScales; i++) {
    std::stringstream sdname;
    sdname << "Strain Difference " << i << " Name";
    strainDifference[i] = Teuchos::rcp(new cHMC2Tensor(
        p.get<std::string>(sdname.str()),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP 2Tensor Data Layout")));
    std::stringstream sdgradname;
    sdgradname << "Micro Strain Gradient " << i << " Name";
    microStrainGradient[i] = Teuchos::rcp(new cHMC3Tensor(
        p.get<std::string>(sdgradname.str()),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP 3Tensor Data Layout")));
    std::stringstream msname;
    msname << "Micro Stress " << i << " Name";
    microStress[i] = Teuchos::rcp(new HMC2Tensor(
        p.get<std::string>(msname.str()),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP 2Tensor Data Layout")));
    std::stringstream dsname;
    dsname << "Double Stress " << i << " Name";
    doubleStress[i] = Teuchos::rcp(new HMC3Tensor(
        p.get<std::string>(dsname.str()),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP 3Tensor Data Layout")));
  }

  lengthScale.resize(numMicroScales);
  betaParameter.resize(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    std::string mySublist                 = Albany::strint("Microscale", i + 1);
    const Teuchos::ParameterList& msModel = p.sublist(mySublist);
    lengthScale[i]   = msModel.get<RealType>("Length Scale");
    betaParameter[i] = msModel.get<RealType>("Beta Constant");
  }

  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP 2Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(strain);
  for (int i = 0; i < numMicroScales; i++) {
    this->addDependentField(*(strainDifference[i]));
    this->addDependentField(*(microStrainGradient[i]));
  }
  this->addEvaluatedField(stress);
  for (int i = 0; i < numMicroScales; i++) {
    this->addEvaluatedField(*(microStress[i]));
    this->addEvaluatedField(*(doubleStress[i]));
  }

  this->setName("Stresses" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
Stresses<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress, fm);
  this->utils.setFieldData(strain, fm);

  int n = this->numMicroScales;
  for (int i = 0; i < n; i++) {
    this->utils.setFieldData(*(strainDifference[i]), fm);
    this->utils.setFieldData(*(microStrainGradient[i]), fm);
    this->utils.setFieldData(*(microStress[i]), fm);
    this->utils.setFieldData(*(doubleStress[i]), fm);
  }
}

//**********************************************************************
/*\begin{text}
   This function computes the stresses assuming a linear elastic response
   \begin{equation}
     \left\{ \begin{array}{c}
     \sigma^p_{ij} \\
     \beta^{np}_{ij} \\
     \bar{\bar{\beta}}^{np}_{ijk} \\
     \end{array} \right\}
      =
     \left[ \begin{array}{ccc}
     C_{ijkl} & 0 & 0 \\
     0 & B_{ijkl} & 0 \\
     0 & 0 & A_{ijklmn} \\
     \end{array} \right]
     \left\{ \begin{array}{c}
     \epsilon^p_{kl} \\
     \epsilon^p_{kl} - \epsilon^{np}_{kl} \\
     \epsilon^{np}_{kl,m}
     \end{array} \right\}
   \end{equation}
   where $C_{ijkl}$ are the linear elastic constants at the macroscale,
$B_{ijkl} = \beta C_{ijkl}$, and $A_{ijklmn} = l^2 \beta C_{ijlm}\delta_{kn}$.
\end{text}*/
template <typename EvalT, typename Traits>
void
Stresses<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  //  ScalarT C11,C33,C12,C23,C44,C66;

  // Irina TOFIX pointers
  TEUCHOS_TEST_FOR_EXCEPT_MSG(
      0 == 0, "Stress:: evaluator has to be fixed for Kokkos data types");
  /*
    switch (numDims) {
    case 1:
      // Compute Stress (uniaxial strain)
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
        for (std::size_t qp=0; qp < numQPs; ++qp)
          stress(cell,qp,0,0) = C11 * strain(cell,qp,0,0);
      break;
    case 2:
      // Compute Stress (plane strain)
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          ScalarT &e1 = strain(cell,qp,0,0), &e2 = strain(cell,qp,1,1), &e3 =
    strain(cell,qp,0,1); stress(cell,qp,0,0) = C11*e1 + C12*e2;
          stress(cell,qp,1,1) = C12*e1 + C11*e2;
          stress(cell,qp,0,1) = C44*e3;
          stress(cell,qp,1,0) = stress(cell,qp,0,1);
        }
      }
      // Compute Micro Stress
      for(int i=0; i<numMicroScales; i++){
        HMC2Tensor &sd = *(strainDifference[i]);
        HMC2Tensor &ms = *(microStress[i]);
        ScalarT beta = betaParameter[i];
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            ScalarT& e1 = sd(cell,qp,0,0), e2 = sd(cell,qp,1,1), e3 =
    sd(cell,qp,0,1), e4 = sd(cell,qp,1,0); ms(cell,qp,0,0) = beta*(C11*e1 +
    C12*e2); ms(cell,qp,1,1) = beta*(C12*e1 + C11*e2); ms(cell,qp,0,1) =
    beta*(C44*e3); ms(cell,qp,1,0) = beta*(C44*e4);
          }
        }
      }
      // Compute Double Stress
      for(int i=0; i<numMicroScales; i++){
        HMC3Tensor &msg = *(microStrainGradient[i]);
        HMC3Tensor &ds = *(doubleStress[i]);
        ScalarT beta = lengthScale[i]*lengthScale[i]*betaParameter[i];
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            for (std::size_t k=0; k < numDims; ++k) {
              ScalarT& e1 = msg(cell,qp,0,0,k), e2 = msg(cell,qp,1,1,k), e3 =
    msg(cell,qp,0,1,k), e4 = msg(cell,qp,1,0,k); ds(cell,qp,0,0,k) =
    beta*(C11*e1 + C12*e2); ds(cell,qp,1,1,k) = beta*(C12*e1 + C11*e2);
              ds(cell,qp,0,1,k) = beta*(C44*e3);
              ds(cell,qp,1,0,k) = beta*(C44*e4);
            }
          }
        }
      }
      break;
    case 3:
      // Compute Stress
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          ScalarT &e1 = strain(cell,qp,0,0), &e2 = strain(cell,qp,1,1), &e3 =
    strain(cell,qp,2,2); ScalarT &e4 = strain(cell,qp,1,2), &e5 =
    strain(cell,qp,0,2), &e6 = strain(cell,qp,0,1); stress(cell,qp,0,0) = C11*e1
    + C12*e2 + C23*e3; stress(cell,qp,1,1) = C12*e1 + C11*e2 + C23*e3;
          stress(cell,qp,2,2) = C23*e1 + C23*e2 + C33*e3;
          stress(cell,qp,1,2) = C44*e4;
          stress(cell,qp,0,2) = C44*e5;
          stress(cell,qp,0,1) = C66*e6;
          stress(cell,qp,1,0) = stress(cell,qp,0,1);
          stress(cell,qp,2,0) = stress(cell,qp,0,2);
          stress(cell,qp,2,1) = stress(cell,qp,1,2);
        }
      }
      // Compute Micro Stress
      for(int i=0; i<numMicroScales; i++){
        HMC2Tensor &sd = *(strainDifference[i]);
        HMC2Tensor &ms = *(microStress[i]);
        ScalarT beta = betaParameter[i];
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            ScalarT& e1 = sd(cell,qp,0,0), e2 = sd(cell,qp,1,1), e3 =
    sd(cell,qp,2,2); ScalarT& e4 = sd(cell,qp,1,2), e5 = sd(cell,qp,0,2), e6 =
    sd(cell,qp,0,1); ScalarT& e7 = sd(cell,qp,2,1), e8 = sd(cell,qp,2,0), e9 =
    sd(cell,qp,1,0); ms(cell,qp,0,0) = beta*(C11*e1 + C12*e2 + C23*e3);
            ms(cell,qp,1,1) = beta*(C12*e1 + C11*e2 + C23*e3);
            ms(cell,qp,2,2) = beta*(C23*e1 + C23*e2 + C33*e3);
            ms(cell,qp,1,2) = beta*(C44*e4);
            ms(cell,qp,0,2) = beta*(C44*e5);
            ms(cell,qp,0,1) = beta*(C66*e6);
            ms(cell,qp,1,0) = beta*(C44*e9);
            ms(cell,qp,2,0) = beta*(C44*e8);
            ms(cell,qp,2,1) = beta*(C66*e7);
          }
        }
      }
      // Compute Double Stress
      for(int i=0; i<numMicroScales; i++){
        HMC3Tensor &msg = *(microStrainGradient[i]);
        HMC3Tensor &ds = *(doubleStress[i]);
        ScalarT beta = lengthScale[i]*lengthScale[i]*betaParameter[i];
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            for (std::size_t k=0; k < numDims; ++k) {
              ScalarT& e1 = msg(cell,qp,0,0,k), e2 = msg(cell,qp,1,1,k), e3 =
    msg(cell,qp,2,2,k); ScalarT& e4 = msg(cell,qp,1,2,k), e5 =
    msg(cell,qp,0,2,k), e6 = msg(cell,qp,0,1,k); ScalarT& e7 =
    msg(cell,qp,2,1,k), e8 = msg(cell,qp,2,0,k), e9 = msg(cell,qp,1,0,k);
              ds(cell,qp,0,0,k) = beta*(C11*e1 + C12*e2 + C23*e3);
              ds(cell,qp,1,1,k) = beta*(C12*e1 + C11*e2 + C23*e3);
              ds(cell,qp,2,2,k) = beta*(C23*e1 + C23*e2 + C33*e3);
              ds(cell,qp,1,2,k) = beta*(C44*e4);
              ds(cell,qp,0,2,k) = beta*(C44*e5);
              ds(cell,qp,0,1,k) = beta*(C66*e6);
              ds(cell,qp,1,0,k) = beta*(C44*e9);
              ds(cell,qp,2,0,k) = beta*(C44*e8);
              ds(cell,qp,2,1,k) = beta*(C66*e7);
            }
          }
        }
      }
      break;
    }
  */
}
//**********************************************************************

}  // namespace HMC
