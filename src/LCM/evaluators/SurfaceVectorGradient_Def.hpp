/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Tensor.h"

namespace LCM {

//**********************************************************************
  template<typename EvalT, typename Traits>
  SurfaceVectorGradient<EvalT, Traits>::
  SurfaceVectorGradient(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl) :
    thickness      (p.get<double>("thickness")), 
    currentBasis   (p.get<std::string>("Current Basis Name"),dl->qp_tensor),
    refDualBasis   (p.get<std::string>("Reference Dual Basis Name"),dl->qp_tensor),
    refNormal      (p.get<std::string>("Reference Normal Name"),dl->qp_vector),
    jump           (p.get<std::string>("Vector Jump Name"),dl->qp_vector),
    gradient       (p.get<std::string>("Surface Vector Gradient Name"),dl->qp_tensor)
  {
    this->addDependentField(currentBasis);
    this->addDependentField(refDualBasis);
    this->addDependentField(refNormal);
    this->addDependentField(jump);

    this->addEvaluatedField(gradient);

    this->setName("Surface Vector Gradient"+PHX::TypeString<EvalT>::value);

    Teuchos::RCP<PHX::DataLayout> nv_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    nv_dl->dimensions(dims);
    worksetSize = dims[0];
    numNodes = dims[1];
    numDims = dims[2];

    Teuchos::RCP<PHX::DataLayout> qpv_dl =
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
    qpv_dl->dimensions(dims);
    numQPs = dims[1];

    numPlaneNodes = numNodes / 2;
    numPlaneDims = numDims - 1;

    // Allocate Temporary FieldContainers
    refValues.resize(numPlaneNodes, numQPs);
    refGrads.resize(numPlaneNodes, numQPs, numPlaneDims);
    refPoints.resize(numQPs, numPlaneDims);
    refWeights.resize(numQPs);

    // Pre-Calculate reference element quantitites
    cubature->getCubature(refPoints, refWeights);
    intrepidBasis->getValues(refValues, refPoints, Intrepid::OPERATOR_VALUE);
    intrepidBasis->getValues(refGrads, refPoints, Intrepid::OPERATOR_GRAD);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceVectorGradient<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(currentBasis,fm);
    this->utils.setFieldData(refDualBasis,fm);
    this->utils.setFieldData(refNormal,fm);
    this->utils.setFieldData(jump,fm);
    this->utils.setFieldData(gradient,fm);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceVectorGradient<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t pt=0; pt < numQPs; ++pt) {
        LCM::Vector<ScalarT> g_0(3, &currentBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> g_1(3, &currentBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> g_2(3, &currentBasis(cell, pt, 2, 0));
        LCM::Vector<ScalarT> G_2(3, &refNormal(cell, pt, 0));
        LCM::Vector<ScalarT> d(3, &jump(cell, pt, 0));
        LCM::Vector<ScalarT> G0(3, &refDualBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> G1(3, &refDualBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> G2(3, &refDualBasis(cell, pt, 2, 0));

        LCM::Tensor<ScalarT> F1(LCM::bun(g_0, G0) + LCM::bun(g_1, G1) + LCM::bun(g_2, G2));
        // for Jay: bun()
        LCM::Tensor<ScalarT> F2((1 / thickness) * LCM::bun(d, G_2));

        LCM::Tensor<ScalarT> F = F1 + F2;

        gradient(cell, pt, 0, 0) = F(0, 0);
        gradient(cell, pt, 0, 1) = F(0, 1);
        gradient(cell, pt, 0, 2) = F(0, 2);
        gradient(cell, pt, 1, 0) = F(1, 0);
        gradient(cell, pt, 1, 1) = F(1, 1);
        gradient(cell, pt, 1, 2) = F(1, 2);
        gradient(cell, pt, 2, 0) = F(2, 0);
        gradient(cell, pt, 2, 1) = F(2, 1);
        gradient(cell, pt, 2, 2) = F(2, 2);
      }
    }
  }
  //**********************************************************************  
}
