//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_RealSpaceTools.hpp"

#include <typeinfo>

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
ThermoMechanicalEnergyResidual<EvalT, Traits>::
ThermoMechanicalEnergyResidual(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout") ),
  Temperature (p.get<std::string>                   ("QP Variable Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  ThermalCond (p.get<std::string>                   ("Thermal Conductivity Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout") ),
  TGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout") ),
  Source      (p.get<std::string>                   ("Source Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  F           (p.get<std::string>                   ("Deformation Gradient Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  mechSource  (p.get<std::string>                   ("Mechanical Source Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  deltaTime   (p.get<std::string>                   ("Delta Time Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout") ),
  density     (p.get<RealType>("Density") ),
  Cv          (p.get<RealType>("Heat Capacity") ),
  TResidual   (p.get<std::string>                   ("Residual Name"),
               p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout") ),
  haveSource  (p.get<bool>("Have Source") )
{
  this->addDependentField(wBF);
  this->addDependentField(Temperature);
  this->addDependentField(ThermalCond);
  this->addDependentField(TGrad);
  this->addDependentField(wGradBF);
  this->addDependentField(F);
  this->addDependentField(mechSource);
  this->addDependentField(deltaTime);
  if (haveSource) this->addDependentField(Source);
  this->addEvaluatedField(TResidual);

  tempName = p.get<std::string>("QP Variable Name")+"_old";

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  worksetSize = dims[0];
  numQPs  = dims[1];
  numDims = dims[2];

  // Allocate workspace
  flux.resize(dims[0], numQPs, numDims);
  C.resize(worksetSize, numQPs, numDims, numDims);
  Cinv.resize(worksetSize, numQPs, numDims, numDims);
  CinvTgrad.resize(worksetSize, numQPs, numDims);
  Tdot.resize(worksetSize, numQPs);

  this->setName("ThermoMechanicalEnergyResidual"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermoMechanicalEnergyResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(Temperature,fm);
  this->utils.setFieldData(ThermalCond,fm);
  this->utils.setFieldData(TGrad,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(F,fm);
  this->utils.setFieldData(mechSource,fm);
  this->utils.setFieldData(deltaTime,fm);
  if (haveSource)  this->utils.setFieldData(Source,fm);

  this->utils.setFieldData(TResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermoMechanicalEnergyResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  bool print = false;
  //if (typeid(ScalarT) == typeid(RealType)) print = true;

  // alias the function space tools
  typedef Intrepid::FunctionSpaceTools FST;

  // get old temperature
  Albany::MDArray Temperature_old = (*workset.stateArrayPtr)[tempName];

  // time step
  ScalarT dt = deltaTime(0);

  // compute the 'material' flux
  FST::tensorMultiplyDataData<ScalarT> (C, F, F, 'T');
  Intrepid::RealSpaceTools<ScalarT>::inverse(Cinv, C);
  FST::tensorMultiplyDataData<ScalarT> (CinvTgrad, Cinv, TGrad);
  FST::scalarMultiplyDataData<ScalarT> (flux, ThermalCond, CinvTgrad);

   FST::integrate<ScalarT>(TResidual, flux, wGradBF, Intrepid::COMP_CPP, false); // "false" overwrites

  if (haveSource) {
    for (int i=0; i<Source.dimension(0); i++) 
       for (int j=0; j<Source.dimension(1); j++) 
          Source(i,j) *= -1.0;
     FST::integrate<ScalarT>(TResidual, Source, wBF, Intrepid::COMP_CPP, true); // "true" sums into
  }

 for (int i=0; i<mechSource.dimension(0); i++) 
       for (int j=0; j<mechSource.dimension(1); j++)
           mechSource(i,j) *= -1.0;
    FST::integrate<ScalarT>(TResidual, mechSource, wBF, Intrepid::COMP_CPP, true); // "true" sums into


//Irina comment: code below was commented out
  //if (workset.transientTerms && enableTransient)
  //   FST::integrate<ScalarT>(TResidual, Tdot, wBF, Intrepid::COMP_CPP, true); // "true" sums into
  //
  // compute factor
  ScalarT fac(0.0);
  if (dt > 0.0)
    fac = ( density * Cv ) / dt;

  for (int cell=0; cell < workset.numCells; ++cell)
    for (int qp=0; qp < numQPs; ++qp)
      Tdot(cell,qp) = fac * ( Temperature(cell,qp) - Temperature_old(cell,qp) );
   FST::integrate<ScalarT>(TResidual, Tdot, wBF, Intrepid::COMP_CPP, true); // "true" sums into

  if (print)
  {
    std::cout << " *** ThermoMechanicalEnergyResidual *** " << std::endl;
    std::cout << "  **   dt: " << dt << std::endl;
    std::cout << "  **  rho: " << density << std::endl;
    std::cout << "  **   Cv: " << Cv << std::endl;
    for (unsigned int cell(0); cell < workset.numCells; ++cell)
    {
      std::cout << "  ** Cell: " << cell << std::endl;
      for (unsigned int qp(0); qp < numQPs; ++qp)
      {
        std::cout << "   * QP: " << std::endl;
       std::cout << "    F   : ";
       for (unsigned int i(0); i < numDims; ++i)
         for (unsigned int j(0); j < numDims; ++j)
           std::cout << F(cell,qp,i,j) << " ";
       std::cout << std::endl;

        std::cout << "    C   : ";
        for (unsigned int i(0); i < numDims; ++i)
          for (unsigned int j(0); j < numDims; ++j)
            std::cout << C(cell,qp,i,j) << " ";
        std::cout << std::endl;

        std::cout << "    T   : " << Temperature(cell,qp) << std::endl;
        std::cout << "    Told: " << Temperature(cell,qp) << std::endl;
        std::cout << "    k   : " << ThermalCond(cell,qp) << std::endl;
      }
    }
  }
}

//**********************************************************************
}
