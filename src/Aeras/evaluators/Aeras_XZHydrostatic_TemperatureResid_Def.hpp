//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "PHAL_Utilities.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_TemperatureResid<EvalT, Traits>::
XZHydrostatic_TemperatureResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF             (p.get<std::string> ("Weighted BF Name"),               dl->node_qp_scalar),
  wGradBF         (p.get<std::string> ("Weighted Gradient BF Name"),      dl->node_qp_gradient),
  temperature     (p.get<std::string> ("QP Temperature"),                 dl->node_scalar_level),
  temperatureGrad (p.get<std::string> ("Gradient QP Temperature"),        dl->qp_gradient_level),
  temperatureDot  (p.get<std::string> ("QP Time Derivative Temperature"), dl->node_scalar_level),
  temperatureSrc  (p.get<std::string> ("Temperature Source"),             dl->qp_scalar_level),
  velocity        (p.get<std::string> ("Velocity"),                       dl->node_vector_level),
  omega           (p.get<std::string> ("Omega"),                          dl->node_scalar_level),
  etadotdT        (p.get<std::string> ("EtaDotdT"),                       dl->qp_scalar_level),
  Residual        (p.get<std::string> ("Residual Name"),                  dl->node_scalar_level),
  viscosity       (p.isParameter("XZHydrostatic Problem") ? 
                   p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("Viscosity", 0.0):
                   p.get<Teuchos::ParameterList*>("Hydrostatic Problem")  ->get<double>("Viscosity", 0.0)),
  Cp              (p.isParameter("XZHydrostatic Problem") ? 
                   p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("Cp", 1005.7):
                   p.get<Teuchos::ParameterList*>("Hydrostatic Problem")->get<double>("Cp", 1005.7)),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numQPs   ( dl->node_qp_scalar          ->dimension(2)),
  numDims  ( dl->node_qp_gradient        ->dimension(3)),
  numLevels( dl->node_scalar_level       ->dimension(2))
{
  Teuchos::ParameterList* xsa_params =
    p.isParameter("XZHydrostatic Problem") ? 
      p.get<Teuchos::ParameterList*>("XZHydrostatic Problem"):
      p.get<Teuchos::ParameterList*>("Hydrostatic Problem");
  Re = xsa_params->get<double>("Reynolds Number", 1.0); //Default: Re=1
  //std::cout << "XZHydrostatic_TemperatureResid: Re= " << Re << std::endl;

  this->addDependentField(temperature);
  this->addDependentField(temperatureGrad);
  this->addDependentField(temperatureDot);
  this->addDependentField(temperatureSrc);
  this->addDependentField(velocity);
  this->addDependentField(omega);
  this->addDependentField(etadotdT);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);

  this->setName("Aeras::XZHydrostatic_TemperatureResid" + PHX::typeAsString<EvalT>());

  // Register Reynolds number as Sacado-ized Parameter
  Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib> >("Parameter Library");
  this->registerSacadoParameter("Reynolds Number", paramLib);

  Prandtl = 0.71;

  pureAdvection = xsa_params->get<bool>("Pure Advection", false);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_TemperatureResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(temperature,    fm);
  this->utils.setFieldData(temperatureGrad,fm);
  this->utils.setFieldData(temperatureDot, fm);
  this->utils.setFieldData(temperatureSrc, fm);
  this->utils.setFieldData(velocity,       fm);
  this->utils.setFieldData(omega,          fm);
  this->utils.setFieldData(etadotdT,       fm);
  this->utils.setFieldData(wBF,            fm);
  this->utils.setFieldData(wGradBF,        fm);

  this->utils.setFieldData(Residual,       fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_TemperatureResid<EvalT, Traits>::
operator() (const XZHydrostatic_TemperatureResid_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node) {
    for (int level=0; level < numLevels; ++level) {
      Residual(cell,node,level)   =  temperatureSrc(cell,node,level)                             *wBF(cell,node,node)
                                  -  omega(cell,node,level)                                      *wBF(cell,node,node)
                                  +  etadotdT(cell,node,level)                                   *wBF(cell,node,node)
                                  +  temperatureDot(cell,node,level)                             *wBF(cell,node,node);

      for (int qp=0; qp < numQPs; ++qp) {
        for (int dim=0; dim < numDims; ++dim) {
          Residual(cell,node,level) += velocity(cell,qp,level,dim)*temperatureGrad(cell,qp,level,dim)*wBF(cell,node,qp);
        }
      }
    }
//     for (int level = 0; level < 2; ++level){
//     	Residual(cell,node,level) += (viscosity/Prandtl)*temperatureGrad(cell,qp,level,dim)*wGradBF(cell,node,qp,dim);
// 	}
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_TemperatureResid<EvalT, Traits>::
operator() (const XZHydrostatic_TemperatureResid_pureAdvection_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node)
    for (int level=0; level < numLevels; ++level)
      Residual(cell,node,level)   = temperatureDot(cell,node,level)*wBF(cell,node,node);
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_TemperatureResid<EvalT, Traits>::
operator() (const XZHydrostatic_TemperatureResid_Laplace_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node) {
    for (int level=0; level < numLevels; ++level) {
      Residual(cell,node,level) = 0;
      for (int qp=0; qp < numQPs; ++qp) {
        for (int dim=0; dim < numDims; ++dim) {
          Residual(cell,node,level) += temperatureGrad(cell,qp,level,dim)*wGradBF(cell,node,qp,dim);
        }
      }
    }
  }
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_TemperatureResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  double j_coeff = workset.j_coeff;
  double n_coeff = workset.n_coeff;
  obtainLaplaceOp = ((n_coeff == 22.0)&&(j_coeff == 1.0)) ? true : false;

//  std::cout <<"In temperature resid: Laplace = " << obtainLaplaceOp << "\n";

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  PHAL::set(Residual, 0.0);
  if ( !obtainLaplaceOp ) {
    if( !pureAdvection ) {
      for (int cell=0; cell < workset.numCells; ++cell) {
        for (int node=0; node < numNodes; ++node) {
          for (int level=0; level < numLevels; ++level) {
            for (int qp=0; qp < numQPs; ++qp) {
              for (int dim=0; dim < numDims; ++dim) {
                Residual(cell,node,level) += velocity(cell,qp,level,dim)*temperatureGrad(cell,qp,level,dim)*wBF(cell,node,qp);
              }
            }
          }
// 		  for (int level = 0; level < 2; ++level) {
//          for (int qp = 0; qp < numQPs; ++qp ) {
//           	for (int dim = 0; dim < numDims; ++dim) {
//           	Residual(cell,node,level) += (viscosity/Prandtl)*temperatureGrad(cell,qp,level,dim)*wGradBF(cell,node,qp,dim);
//           	}
//           }
//         }
        }

        for (int qp=0; qp < numQPs; ++qp) {
          int node = qp;
          for (int level=0; level < numLevels; ++level) {
            Residual(cell,node,level)   += temperatureSrc(cell,qp,level)                             *wBF(cell,node,qp)
                                        -  omega(cell,qp,level)                                      *wBF(cell,node,qp)
                                        +  etadotdT(cell,qp,level)                                   *wBF(cell,node,qp)
                                        +  temperatureDot(cell,qp,level)                             *wBF(cell,node,qp);
          }
        }
      }

      /*//OG debugging statements
      {
      int level = 2;
      std::cout << "level = " << level << " PRINT TRESID ------------------ \n";
      for (int qp=0; qp < numQPs; ++qp) {
        std::cout << "tresid qp = "<<qp << "value = " <<  Residual(23,qp,level)/wBF(23,qp,qp) << "\n";
      }
      }*/
    }//end of (if not pureAdvection)

    else {
      for (int cell=0; cell < workset.numCells; ++cell)
        for (int node=0; node < numNodes; ++node)
          for (int level=0; level < numLevels; ++level)
            Residual(cell,node,level)   += temperatureDot(cell,node,level)*wBF(cell,node,node);
    }
  }//end of (if not Laplace op)

  else {//building Laplace
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
        for (int level=0; level < numLevels; ++level) {
          for (int qp=0; qp < numQPs; ++qp) {
            for (int dim=0; dim < numDims; ++dim) {
              Residual(cell,node,level) += temperatureGrad(cell,qp,level,dim)*wGradBF(cell,node,qp,dim);
            }
          }
        }
      }
    }
  }

#else
  if ( !obtainLaplaceOp ) {
    if( !pureAdvection ) {
      Kokkos::parallel_for(XZHydrostatic_TemperatureResid_Policy(0,workset.numCells),*this);
    }

    else {
      Kokkos::parallel_for(XZHydrostatic_TemperatureResid_pureAdvection_Policy(0,workset.numCells),*this);
    }
  }

  else {
    Kokkos::parallel_for(XZHydrostatic_TemperatureResid_Laplace_Policy(0,workset.numCells),*this);
  }

#endif
}

//**********************************************************************
// Provide Access to Parameter for sensitivity/optimization/UQ
template<typename EvalT,typename Traits>
typename XZHydrostatic_TemperatureResid<EvalT,Traits>::ScalarT&
XZHydrostatic_TemperatureResid<EvalT,Traits>::getValue(const std::string &n)
{
  return Re;
}

}
