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
  SurfaceScalarGradient<EvalT, Traits>::
  SurfaceScalarGradient(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl) :
    thickness      (p.get<double>("thickness")), 
    cubature       (p.get<Teuchos::RCP<Intrepid::Cubature<RealType> > >("Cubature")), 
    currentBasis   (p.get<std::string>("Current Basis Name"),dl->qp_tensor),
    refDualBasis   (p.get<std::string>("Reference Dual Basis Name"),dl->qp_tensor),
    refNormal      (p.get<std::string>("Reference Normal Name"),dl->qp_vector),
    jump           (p.get<std::string>("Vector Jump Name"),dl->qp_scalar),
    scalarGrad        (p.get<std::string>("Surface Vector Gradient Name"),dl->qp_vector),
   // J              (p.get<std::string>("Surface Vector Gradient Determinant Name"),dl->qp_scalar),
    weights        (p.get<std::string>("Weights Name"),dl->qp_scalar),
    weightedAverage(false),
    alpha(0.0)
  {
    if ( p.isType<string>("Weighted Volume Average J Name") )
      weightedAverage = p.get<bool>("Weighted Volume Average J");
    if ( p.isType<double>("Average J Stabilization Parameter Name") )
      alpha = p.get<double>("Average J Stabilization Parameter");

    this->addDependentField(currentBasis);
    this->addDependentField(refDualBasis);
    this->addDependentField(refNormal);
    this->addDependentField(jump);

    this->addEvaluatedField(scalarGrad);
  //  this->addEvaluatedField(J);

    this->setName("Surface Vector Gradient"+PHX::TypeString<EvalT>::value);

    std::vector<PHX::DataLayout::size_type> dims;
    dl->node_vector->dimensions(dims);
    worksetSize = dims[0];
    numNodes = dims[1];
    numDims = dims[2];

    numQPs = cubature->getNumPoints();

    numPlaneNodes = numNodes / 2;
    numPlaneDims = numDims - 1;
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceScalarGradient<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(currentBasis,fm);
    this->utils.setFieldData(refDualBasis,fm);
    this->utils.setFieldData(refNormal,fm);
    this->utils.setFieldData(jump,fm);
    this->utils.setFieldData(scalarGrad,fm);
  //  this->utils.setFieldData(J,fm);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void SurfaceScalarGradient<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t pt=0; pt < numQPs; ++pt) {
        LCM::Vector<ScalarT> g_0(3, &currentBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> g_1(3, &currentBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> g_2(3, &currentBasis(cell, pt, 2, 0));
        LCM::Vector<ScalarT> G_2(3, &refNormal(cell, pt, 0));
        LCM::Vector<ScalarT> G0(3, &refDualBasis(cell, pt, 0, 0));
        LCM::Vector<ScalarT> G1(3, &refDualBasis(cell, pt, 1, 0));
        LCM::Vector<ScalarT> G2(3, &refDualBasis(cell, pt, 2, 0));


        // for now, there is no normal contribution, I will add it later on --Sun
        LCM::Vector<ScalarT> F2((1 / thickness) * jump(cell, pt) * G_2); // orthogonal contribution

        LCM::Vector<ScalarT> F = F2;

        scalarGrad(cell, pt, 0) =F(0);
        scalarGrad(cell, pt, 1) =F(1);
        scalarGrad(cell, pt, 2) =F(2);


      }
    }

    /*
    if (weightedAverage)
    {
      ScalarT Jbar, wJbar, vol;
      for (std::size_t cell=0; cell < workset.numCells; ++cell)
      {
        Jbar = 0.0;
        vol = 0.0;
        for (std::size_t qp=0; qp < numQPs; ++qp)
        {
          Jbar += weights(cell,qp) * std::log( J(cell,qp) );
          vol  += weights(cell,qp);
        }
        Jbar /= vol;

        // Jbar = std::exp(Jbar);
        for (std::size_t qp=0; qp < numQPs; ++qp)
        {
          for (std::size_t i=0; i < numDims; ++i)
          {
            for (std::size_t j=0; j < numDims; ++j)
            {
              wJbar = std::exp( (1-alpha) * Jbar + alpha * std::log( J(cell,qp) ) );
              scalarGrad(cell,qp,i,j) *= std::pow( wJbar / J(cell,qp) ,1./3. );
            }
          }
          J(cell,qp) = wJbar;
        }
      }
    }

    */

  }
  //**********************************************************************  
}
