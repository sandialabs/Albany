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

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Eta.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_Pressure<EvalT, Traits>::
XZHydrostatic_Pressure(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  Ps        (p.get<std::string> ("Pressure Level 0"), dl->node_scalar),
  Pressure  (p.get<std::string> ("Pressure"),         dl->node_scalar_level),
  Pi        (p.get<std::string> ("Pi"),               dl->node_scalar_level),

  numNodes ( dl->node_scalar          ->dimension(1)),
  numLevels( dl->node_scalar_level    ->dimension(2)),
  E (Eta<EvalT>::self())
{

  Teuchos::ParameterList* xzhydrostatic_params =
    p.isParameter("XZHydrostatic Problem") ? 
      p.get<Teuchos::ParameterList*>("XZHydrostatic Problem"):
      p.get<Teuchos::ParameterList*>("Hydrostatic Problem");

  this->addDependentField(Ps);

  this->addEvaluatedField(Pressure);
  this->addEvaluatedField(Pi);
  this->setName("Aeras::XZHydrostatic_Pressure" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Pressure<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Ps       ,fm);
  this->utils.setFieldData(Pressure ,fm);
  this->utils.setFieldData(Pi       ,fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_Pressure<EvalT, Traits>::
operator() (const XZHydrostatic_Pressure_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node) {
    for (int level=0; level < numLevels; ++level) {
      Pressure(cell,node,level) = E.A(level)*E.p0() + E.B(level)*Ps(cell,node);
    }
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_Pressure<EvalT, Traits>::
operator() (const XZHydrostatic_Pressure_Pi_Tag& tag, const int& cell) const{
  for (int node=0; node < numNodes; ++node) {
    for (int level=0; level < numLevels; ++level) {
      const ScalarT pm   = level             ? 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level-1) ) : E.ptop();
      const ScalarT pp   = level<numLevels-1 ? 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level+1) ) : ScalarT(Ps(cell,node));
      Pi(cell,node,level) = (pp - pm) /E.delta(level);
    }
  }
}
#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Pressure<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      /*//OG debugging statements
      if (cell == 0 && node == 0) {
        std::cout << "Etatop = " << E.etatop() <<"\n";
    	for (int level=0; level < numLevels; ++level) {
    	  std::cout << "Here we are level, eta " << level << " " << E.eta(level) << "\n";
    	}
    	for (int level=0; level < numLevels; ++level) {
    	  std::cout << "Here we are A, B " << level << " " << E.A(level) << "  " << E.B(level) << "\n";
    	}
      }
      */
      for (int level=0; level < numLevels; ++level) {
        Pressure(cell,node,level) = E.A(level)*E.p0() + E.B(level)*Ps(cell,node);
        //std::cout <<"In Pressure "<< " Ps" << Ps(cell,node) <<" workset time" << workset.current_time << "\n";
        //if (cell == 0 && node == 0) {
        //  std::cout <<"In Pressure "<< "level: " << level << "  " 
        //                            << "A(level): " << E.A(level) << "  " 
        //                            << "B(level): " << E.B(level) << "  " 
        //                            << "eta(level): " << E.eta(level) << "\n";
        //}
      }
      //here instead of computing eta, A, B, and pressure at level interfaces directly,
      //averages are used to approx. pressure at level interfaces.
      for (int level=0; level < numLevels; ++level) {
        //OG Why not analyt. relationship? Verify this in homme. Update: Homme uses averaging below because of consistency restrictions as in Chapter 12.
        //That is, it is required that B (and A) coeffs at midpoints equal averages from closest interfaces.
        //const ScalarT pm   = E.A(level-.5)*E.p0() + E.B(level-.5)*Ps(cell,node);
        //const ScalarT pp   = E.A(level+.5)*E.p0() + E.B(level+.5)*Ps(cell,node);
        const ScalarT pm   = level             ? 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level-1) ) : E.ptop();
        const ScalarT pp   = level<numLevels-1 ? 0.5*( Pressure(cell,node,level) + Pressure(cell,node,level+1) ) : ScalarT(Ps(cell,node));
        Pi(cell,node,level) = (pp - pm) /E.delta(level);
      }
    }
  }

#else
  Kokkos::parallel_for(XZHydrostatic_Pressure_Policy(0,workset.numCells),*this);
  Kokkos::parallel_for(XZHydrostatic_Pressure_Pi_Policy(0,workset.numCells),*this);

#endif
}
}
