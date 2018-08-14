//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

//#define HARD_CODED_BODY_FORCE_ELASTICITY_RESID

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
ElasticityResid<EvalT, Traits>::ElasticityResid(Teuchos::ParameterList& p)
    : Stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      wGradBF(
          p.get<std::string>("Weighted Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      ExResidual(
          p.get<std::string>("Residual Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node Vector Data Layout"))
{
  this->addDependentField(Stress);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(ExResidual);

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else
    enableTransient = true;

  hasDensity = false;

  if (enableTransient) {
    // Additional fields required for transient capability

    if (p.isParameter("Density Name")) {
      hasDensity = true;
      Teuchos::RCP<PHX::DataLayout> cell_scalar_dl =
          p.get<Teuchos::RCP<PHX::DataLayout>>("Cell Scalar Data Layout");
      density =
          decltype(density)(p.get<std::string>("Density Name"), cell_scalar_dl);
      this->addDependentField(density);
    }

    Teuchos::RCP<PHX::DataLayout> vector_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
    uDotDot = decltype(uDotDot)(
        p.get<std::string>("Time Dependent Variable Name"), vector_dl);
    this->addDependentField(uDotDot);
  }

#ifdef HARD_CODED_BODY_FORCE_ELASTICITY_RESID
  Teuchos::RCP<PHX::DataLayout> node_qp_scalar_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout");
  wBF =
      decltype(wBF)(p.get<std::string>("Weighted BF Name"), node_qp_scalar_dl);
  this->addDependentField(wBF);
#else
  if (enableTransient) {
    Teuchos::RCP<PHX::DataLayout> node_qp_scalar_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout");
    wBF = decltype(wBF)(
        p.get<std::string>("Weighted BF Name"), node_qp_scalar_dl);
    this->addDependentField(wBF);
  }
#endif

  this->setName("ElasticityResid" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
ElasticityResid<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Stress, fm);
  this->utils.setFieldData(wGradBF, fm);

  this->utils.setFieldData(ExResidual, fm);

  if (enableTransient) this->utils.setFieldData(uDotDot, fm);

#ifdef HARD_CODED_BODY_FORCE_ELASTICITY_RESID
  this->utils.setFieldData(wBF, fm);
#else
  if (enableTransient) this->utils.setFieldData(wBF, fm);
#endif

  if (hasDensity) this->utils.setFieldData(density, fm);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
ElasticityResid<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      for (int dim = 0; dim < numDims; dim++) {
        ExResidual(cell, node, dim) = 0.0;
      }
      for (int qp = 0; qp < numQPs; ++qp) {
        for (int i = 0; i < numDims; i++) {
          for (int dim = 0; dim < numDims; dim++) {
            ExResidual(cell, node, i) +=
                Stress(cell, qp, i, dim) * wGradBF(cell, node, qp, dim);
          }
        }
      }
    }
  }

#ifdef HARD_CODED_BODY_FORCE_ELASTICITY_RESID
  std::vector<double> body_force(3);
  body_force[0] = 5.1732283464566922;
  body_force[1] = 0.0;
  body_force[2] = 0.0;
  std::cout << "****WARNING hard-coded body force being applied!  Body force "
               "density = ("
            << body_force[0] << ", " << body_force[1] << ", " << body_force[2]
            << ")" << std::endl;
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      for (int qp = 0; qp < numQPs; ++qp) {
        for (int i = 0; i < numDims; i++) {
          ExResidual(cell, node, i) += body_force[i] * wBF(cell, node, qp);
        }
      }
    }
  }
#endif

  if (workset.transientTerms && enableTransient) {
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < numNodes; ++node) {
        for (int qp = 0; qp < numQPs; ++qp) {
          for (int i = 0; i < numDims; i++) {
            if (hasDensity) {
              ExResidual(cell, node, i) +=
                  density(cell) * uDotDot(cell, qp, i) * wBF(cell, node, qp);
            } else {
              ExResidual(cell, node, i) +=
                  uDotDot(cell, qp, i) * wBF(cell, node, qp);
            }
          }
        }
      }
    }
  }

  //   FST::integrate(ExResidual.get_view(), Stress.get_view(),
  //   wGradBF.get_view(), false); // "false" overwrites
}

//**********************************************************************
}  // namespace LCM
