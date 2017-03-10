//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"
#include "Albany_Utils.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Eta.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_SPressureResid<EvalT, Traits>::
XZHydrostatic_SPressureResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"),                 dl->node_qp_scalar),
  spDot    (p.get<std::string> ("Pressure QP Time Derivative Variable Name"), dl->node_scalar),
  divpivelx(p.get<std::string>("Divergence QP PiVelx"),              dl->qp_scalar_level),
  Residual (p.get<std::string> ("Residual Name"),                    dl->node_scalar),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numQPs   ( dl->node_qp_scalar          ->dimension(2)),
  numLevels( dl->node_scalar_level       ->dimension(2)),
  E (Eta<EvalT>::self())
{

  Teuchos::ParameterList* xsa_params =
	  p.isParameter("XZHydrostatic Problem") ?
	  p.get<Teuchos::ParameterList*>("XZHydrostatic Problem"):
	  p.get<Teuchos::ParameterList*>("Hydrostatic Problem");

  this->addDependentField(wBF);
  this->addDependentField(spDot);
  this->addDependentField(divpivelx);

  this->addEvaluatedField(Residual);

  this->setName("Aeras::XZHydrostatic_SPressureResid" +PHX::typeAsString<EvalT>());

  sp0 = 0.0;

  pureAdvection = xsa_params->get<bool>("Pure Advection", false);

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  delta = E.delta_kokkos;
#endif
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_SPressureResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,      fm);
  this->utils.setFieldData(spDot,    fm);
  this->utils.setFieldData(divpivelx,fm);

  this->utils.setFieldData(Residual, fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_SPressureResid<EvalT, Traits>::
operator() (const int cell, const int qp) const{
    ScalarT sum = 0;
    for (int level=0; level<numLevels; ++level)
	sum += divpivelx(cell,qp,level) * delta(level);
   int node = qp;
   Residual(cell,node) += (spDot(cell,qp) + sum)*wBF(cell,node,qp);
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_SPressureResid<EvalT, Traits>::
operator() (const XZHydrostatic_SPressureResid_pureAdvection_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node)
    Residual(cell,node) += spDot(cell,node)*wBF(cell,node,node);
}
#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_SPressureResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  double j_coeff = workset.j_coeff;
  double n_coeff = workset.n_coeff;
  obtainLaplaceOp = ((n_coeff == 22.0)&&(j_coeff == 1.0)) ? true : false;

  PHAL::set(Residual, 0.0);

//  std::cout <<"In surf pressure resid: Laplace = " << obtainLaplaceOp << "\n";

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if( !obtainLaplaceOp ) {
    if( !pureAdvection ){
      for (int cell=0; cell < workset.numCells; ++cell) {
        for (int qp=0; qp < numQPs; ++qp) {
          ScalarT sum = 0;
	  for (int level=0; level<numLevels; ++level)  sum += divpivelx(cell,qp,level) * E.delta(level);
	  int node = qp;
	  Residual(cell,node) += (spDot(cell,qp) + sum)*wBF(cell,node,qp);
	}
        /*//OG debugging statements
        if(cell == 23){
          std::cout << "Name? " << this->getName() << "-----------------------------------------------------\n";
	  std::cout << "SP residual = " << Residual(cell,0)/wBF(cell,0,0) << " spdot = "<< spDot(cell,0) <<"\n";
	  //std::cout << "SP ITSELF = " << Residual(cell,0) << " spdot = "<< spDot(cell,0) <<"\n";
	}
	*/
      }
      /*//OG debugging statements
      {
        for (int qp=0; qp < numQPs; ++qp) {
          std::cout << "SP resid qp = "<<qp << "value = " <<  Residual(23,qp)/wBF(23,qp,qp) << "\n";
        }
      }*/
    }//end of (if not  pureAdvection)
    else {
      for (int cell=0; cell < workset.numCells; ++cell)
        for (int node=0; node < numNodes; ++node)
  	  Residual(cell,node) += spDot(cell,node)*wBF(cell,node,node);
    }
  }//end of (if build laplace)

  else {
    //no Laplace for surface pressure, zero block instead
  }

#else
  if( !obtainLaplaceOp ) {
    if( !pureAdvection ) {
      XZHydrostatic_SPressureResid_Policy range(          
	{0,0}, {(int)workset.numCells,(int)numQPs}, XZHydrostatic_SPressureResid_TileSize);
      Kokkos::Experimental::md_parallel_for(range,*this);
      cudaCheckError();
    }

    else {
      Kokkos::parallel_for(XZHydrostatic_SPressureResid_pureAdvection_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

  else {
    //no Laplace for surface pressure, zero block instead
  }

#endif
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename XZHydrostatic_SPressureResid<EvalT,Traits>::ScalarT& 
XZHydrostatic_SPressureResid<EvalT,Traits>::getValue(const std::string &n)
{
  if (n=="SPressure") return sp0;
  return sp0;
}

}
