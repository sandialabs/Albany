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
#include "Albany_Utils.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"
#include "Aeras_Dimension.hpp"

#include "Aeras_Eta.hpp"
#include "Kokkos_Vector.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_EtaDotPi<EvalT, Traits>::
XZHydrostatic_EtaDotPi(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  divpivelx      (p.get<std::string> ("Divergence QP PiVelx"),  dl->qp_scalar_level),
  pdotP0         (p.get<std::string> ("Pressure Dot Level 0"),  dl->node_scalar),
  Pi             (p.get<std::string> ("Pi"),                    dl->qp_scalar_level),
  Temperature    (p.get<std::string> ("QP Temperature"),        dl->node_scalar_level),
  Velocity       (p.get<std::string> ("Velocity"),              dl->node_vector_level),
  tracerNames    (p.get< Teuchos::ArrayRCP<std::string> >("Tracer Names")),
  //etadotdtracerNames    (p.get< Teuchos::ArrayRCP<std::string> >("Tracer EtaDotd Names")),
  dedotpitracerdeNames    (p.get< Teuchos::ArrayRCP<std::string> >("Tracer EtaDotd Names")),
  etadotdT       (p.get<std::string> ("EtaDotdT"),              dl->qp_scalar_level),
  etadot         (p.get<std::string> ("EtaDot"),                dl->qp_scalar_level),
  etadotdVelx    (p.get<std::string> ("EtaDotdVelx"),           dl->node_vector_level),
  Pidot          (p.get<std::string> ("PiDot"),                 dl->qp_scalar_level),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numLevels  (dl->node_scalar_level       ->dimension(2)),
  E (Eta<EvalT>::self())
{

  Teuchos::ParameterList* xzhydrostatic_params =
    p.isParameter("XZHydrostatic Problem") ? 
      p.get<Teuchos::ParameterList*>("XZHydrostatic Problem"):
      p.get<Teuchos::ParameterList*>("Hydrostatic Problem");

  this->addDependentField(divpivelx);
  this->addDependentField(pdotP0);
  this->addDependentField(Pi);
  this->addDependentField(Temperature);
  this->addDependentField(Velocity);

  this->addEvaluatedField(etadotdT);
  this->addEvaluatedField(etadot);
  this->addEvaluatedField(etadotdVelx);
  this->addEvaluatedField(Pidot);

  for (int i = 0; i < tracerNames.size(); ++i) {
    PHX::MDField<ScalarT,Cell,QuadPoint,Level> in   (tracerNames[i],          dl->qp_scalar_level);
    //PHX::MDField<ScalarT,Cell,QuadPoint,Level> out  (etadotdtracerNames[i],   dl->qp_scalar_level);
    PHX::MDField<ScalarT,Cell,QuadPoint,Level> out  (dedotpitracerdeNames[i],   dl->qp_scalar_level);
    Tracer[tracerNames[i]]         = in;
    //etadotdTracer[tracerNames[i]] = out;
    dedotpiTracerde[tracerNames[i]] = out;
    this->addDependentField(Tracer[tracerNames[i]]);
    //this->addEvaluatedField(etadotdTracer[tracerNames[i]]);
    this->addEvaluatedField(dedotpiTracerde[tracerNames[i]]);
  }

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Tracer_kokkos.resize(tracerNames.size());
  //etadotdTracer_kokkos.resize(tracerNames.size());
  dedotpiTracerde_kokkos.resize(tracerNames.size());
#endif

  this->setName("Aeras::XZHydrostatic_EtaDotPi"+PHX::typeAsString<EvalT>());

  pureAdvection = xzhydrostatic_params->get<bool>("Pure Advection", false);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_EtaDotPi<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(divpivelx  ,   fm);
  this->utils.setFieldData(pdotP0     ,   fm);
  this->utils.setFieldData(Pi         ,   fm);
  this->utils.setFieldData(Temperature,   fm);
  this->utils.setFieldData(Velocity   ,   fm);
  this->utils.setFieldData(etadotdT   ,   fm);
  this->utils.setFieldData(etadot     ,   fm);
  this->utils.setFieldData(etadotdVelx,   fm);
  this->utils.setFieldData(Pidot,         fm);

  for (int i = 0; i < Tracer.size();  ++i)       this->utils.setFieldData(Tracer[tracerNames[i]], fm);
  //for (int i = 0; i < etadotdTracer.size(); ++i) this->utils.setFieldData(etadotdTracer[tracerNames[i]],fm);
  for (int i = 0; i < dedotpiTracerde.size(); ++i) this->utils.setFieldData(dedotpiTracerde[tracerNames[i]],fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_EtaDotPi<EvalT, Traits>::
operator() (const XZHydrostatic_EtaDotPi_Tag& tag, const int& cell) const{
  for (int qp=0; qp < numQPs; ++qp) {
    ScalarT pdotp0 = 0;
    for (int level=0; level < numLevels; ++level) pdotp0 -= divpivelx(cell,qp,level) * E.delta(level);

    //etadotpi(level) shifted by 1/2
    Kokkos::DynRankView<ScalarT, PHX::Device> etadotpi = Kokkos::createDynRankView(Pidot.get_view(), "etadotpi", numLevels+1);
    for (int level=0; level < numLevels; ++level) {
      //define etadotpi on interfaces
      ScalarT integral = 0;
      for (int j=0; j<=level; ++j) integral += divpivelx(cell,qp,j) * E.delta(j);

      etadotpi(level) = -E.B(level+.5)*pdotp0 - integral;
    }
    etadotpi(0) = etadotpi(numLevels) = 0;

    //Vertical Finite Differencing
    for (int level=0; level < numLevels; ++level) {
      const ScalarT factor     = 1.0/(2.0*Pi(cell,qp,level)*E.delta(level));
      const int level_m = level             ? level-1 : 0;
      const int level_p = level+1<numLevels ? level+1 : level;
      const ScalarT etadotpi_m = etadotpi(level  );
      const ScalarT etadotpi_p = etadotpi(level+1);

      const ScalarT dT_m       = Temperature(cell,qp,level)   - Temperature(cell,qp,level_m);
      const ScalarT dT_p       = Temperature(cell,qp,level_p) - Temperature(cell,qp,level);
      etadotdT(cell,qp,level) = factor * ( etadotpi_p*dT_p + etadotpi_m*dT_m );

      for (int dim=0; dim<numDims; ++dim) {
        const ScalarT dVx_m      = Velocity(cell,qp,level,dim)   - Velocity(cell,qp,level_m,dim);
        const ScalarT dVx_p      = Velocity(cell,qp,level_p,dim) - Velocity(cell,qp,level,dim);
        etadotdVelx(cell,qp,level,dim) = factor * ( etadotpi_p*dVx_p + etadotpi_m*dVx_m );
      }

      //OG: Why for tracers (etaDot delta_eta) operator is different than for velocity, T, etc.?
      for (int i = 0; i < tracerNames.size(); ++i) {
        const ScalarT q_m = 0.5*( d_Tracer[i](cell,qp,level)   / Pi(cell,qp,level)
                          + d_Tracer[i](cell,qp,level_m) / Pi(cell,qp,level_m) );
        const ScalarT q_p = 0.5*( d_Tracer[i](cell,qp,level_p) / Pi(cell,qp,level_p)
                          + d_Tracer[i](cell,qp,level)   / Pi(cell,qp,level)   );
        //d_etadotdTracer[i](cell,qp,level) = ( etadotpi_p*q_p - etadotpi_m*q_m ) / E.delta(level);
        d_dedotpiTracerde[i](cell,qp,level) = ( etadotpi_p*q_p - etadotpi_m*q_m ) / E.delta(level);
      }

      //OG: A tracer eqn for pi, or for q=1. Not relevant for basic hydrostatic version.
      Pidot(cell,qp,level) = - divpivelx(cell,qp,level) - (etadotpi_p - etadotpi_m)/E.delta(level);
    }
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_EtaDotPi<EvalT, Traits>::
operator() (const XZHydrostatic_EtaDotPi_pureAdvection_Tag& tag, const int& cell) const{
  for (int qp=0; qp < numQPs; ++qp) {
    //etadotpi(level) shifted by 1/2
    Kokkos::DynRankView<ScalarT, PHX::Device> etadotpi = Kokkos::createDynRankView(Pidot.get_view(), "etadotpi", numLevels+1);
    for (int level=0; level < numLevels-1; ++level) {
      const ScalarT etadotpi_m = etadot(cell,qp,level  )*Pi(cell,qp,level  );
      const ScalarT etadotpi_p = etadot(cell,qp,level+1)*Pi(cell,qp,level+1);

      //Simple average in vertical direction for etadotpi(level+1/2) at interfaces
      etadotpi(level) = 0.5*(etadotpi_m + etadotpi_p);  
    }
    etadotpi(0) = etadotpi(numLevels) = 0;
        
    //Vertical Finite Differencing
    for (int level=0; level < numLevels; ++level) {
      const ScalarT factor     = 1.0/(2.0*Pi(cell,qp,level)*E.delta(level));
      const int level_m = level             ? level-1 : 0;
      const int level_p = level+1<numLevels ? level+1 : level;
      const ScalarT etadotpi_m = etadotpi(level  );
      const ScalarT etadotpi_p = etadotpi(level+1);

      const ScalarT dT_m       = Temperature(cell,qp,level)   - Temperature(cell,qp,level_m);
      const ScalarT dT_p       = Temperature(cell,qp,level_p) - Temperature(cell,qp,level);
      etadotdT(cell,qp,level) = factor * ( etadotpi_p*dT_p + etadotpi_m*dT_m );

      for (int dim=0; dim<numDims; ++dim) {
        const ScalarT dVx_m      = Velocity(cell,qp,level,dim)   - Velocity(cell,qp,level_m,dim);
        const ScalarT dVx_p      = Velocity(cell,qp,level_p,dim) - Velocity(cell,qp,level,dim);
        etadotdVelx(cell,qp,level,dim) = factor * ( etadotpi_p*dVx_p + etadotpi_m*dVx_m );
      }

      for (int i = 0; i < tracerNames.size(); ++i) {
        const ScalarT q_m = 0.5*( d_Tracer[i](cell,qp,level)   / Pi(cell,qp,level)
                          + d_Tracer[i](cell,qp,level_m) / Pi(cell,qp,level_m) );
        const ScalarT q_p = 0.5*( d_Tracer[i](cell,qp,level_p) / Pi(cell,qp,level_p)
                          + d_Tracer[i](cell,qp,level)   / Pi(cell,qp,level)   );
        d_dedotpiTracerde[i](cell,qp,level) = ( etadotpi_p*q_p - etadotpi_m*q_m ) / E.delta(level);
      }

      Pidot(cell,qp,level) = - divpivelx(cell,qp,level) - (etadotpi_p - etadotpi_m)/E.delta(level);
    }
  }
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_EtaDotPi<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  //etadotpi(level) shifted by 1/2
  std::vector<ScalarT> etadotpi(numLevels+1);

  if (!pureAdvection) 
  {
    /*//OG debugging statements
    std::cout << "Printing DIVPIVELX ----------------------------------------\n";
    for (int level=0; level < numLevels; ++level) {
      std::cout << "level = " << level << "\n";
      for (int qp=0; qp < numQPs; ++qp) {
        std::cout << "qp = "<<qp << " divdp = "<< divpivelx(23,qp,level)*E.delta(level) << " Edelta= "<< E.delta(level) <<"\n";
      }
    }
    */

    //TMS debug
    //for (int level=0; level < numLevels; ++level) {
    //  std::cout << "In EtaDotPi level: " << level << " " 
    //            << "B(level+1/2): " << E.B(level+.5) << std::endl;
    //}

    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {

        ScalarT pdotp0 = 0;
	for (int level=0; level < numLevels; ++level) pdotp0 -= divpivelx(cell,qp,level) * E.delta(level);
	for (int level=0; level < numLevels; ++level) {
	  //define etadotpi on interfaces
	  ScalarT integral = 0;
	  for (int j=0; j<=level; ++j) integral += divpivelx(cell,qp,j) * E.delta(j);
	  etadotpi[level] = -E.B(level+.5)*pdotp0 - integral;
	}
	etadotpi[0] = etadotpi[numLevels] = 0;

	/*//OG debugging statements
	if ((cell == 23) && (qp == 0)) {
	  std::cout << "Etadotdpdn ------------------------------------\n";
	  for (int level=0; level < numLevels+1; ++level) {
	    std::cout << "level = " << level << "etadotdpdn = "<< etadotpi[level] <<"\n";
	  }
	}*/
	//Vertical Finite Differencing
	for (int level=0; level < numLevels; ++level) {
	  const ScalarT factor     = 1.0/(2.0*Pi(cell,qp,level)*E.delta(level));
	  const int level_m = level             ? level-1 : 0;
	  const int level_p = level+1<numLevels ? level+1 : level;
	  const ScalarT etadotpi_m = etadotpi[level  ];
	  const ScalarT etadotpi_p = etadotpi[level+1];

	  const ScalarT dT_m       = Temperature(cell,qp,level)   - Temperature(cell,qp,level_m);
	  const ScalarT dT_p       = Temperature(cell,qp,level_p) - Temperature(cell,qp,level);
	  etadotdT(cell,qp,level) = factor * ( etadotpi_p*dT_p + etadotpi_m*dT_m );

   	  for (int dim=0; dim<numDims; ++dim) {
 	    const ScalarT dVx_m      = Velocity(cell,qp,level,dim)   - Velocity(cell,qp,level_m,dim);
	    const ScalarT dVx_p      = Velocity(cell,qp,level_p,dim) - Velocity(cell,qp,level,dim);
	    etadotdVelx(cell,qp,level,dim) = factor * ( etadotpi_p*dVx_p + etadotpi_m*dVx_m );
	  }
	  //OG: Why for tracers (etaDot delta_eta) operator is different than for velocity, T, etc.?
	  //
	  for (int i = 0; i < tracerNames.size(); ++i) {
	    const ScalarT q_m = 0.5*( Tracer[tracerNames[i]](cell,qp,level)   / Pi(cell,qp,level)
	 		      + Tracer[tracerNames[i]](cell,qp,level_m) / Pi(cell,qp,level_m) );
	    const ScalarT q_p = 0.5*( Tracer[tracerNames[i]](cell,qp,level_p) / Pi(cell,qp,level_p)
	 		      + Tracer[tracerNames[i]](cell,qp,level)   / Pi(cell,qp,level)   );
	    //etadotdTracer[tracerNames[i]](cell,qp,level) = ( etadotpi_p*q_p - etadotpi_m*q_m ) / E.delta(level);
	    dedotpiTracerde[tracerNames[i]](cell,qp,level) = ( etadotpi_p*q_p - etadotpi_m*q_m ) / E.delta(level);
	  }
          //OG: A tracer eqn for pi, or for q=1. Not relevant for basic hydrostatic version.
  	  Pidot(cell,qp,level) = - divpivelx(cell,qp,level) - (etadotpi_p - etadotpi_m)/E.delta(level);
	}

      }
    }
  }//end of (not pure Advection)

  //pure advection: there are many auxiliary variables.
  else {
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {

	for (int level=0; level < numLevels-1; ++level) {
	  const ScalarT etadotpi_m = etadot(cell,qp,level  )*Pi(cell,qp,level  );
	  const ScalarT etadotpi_p = etadot(cell,qp,level+1)*Pi(cell,qp,level+1);
          //Simple average in vertical direction for etadotpi(level+1/2) at interfaces
          etadotpi[level] = 0.5*(etadotpi_m + etadotpi_p);  
        }
	etadotpi[0] = etadotpi[numLevels] = 0;
        
        //Vertical Finite Differencing
        for (int level=0; level < numLevels; ++level) {
	  const ScalarT factor     = 1.0/(2.0*Pi(cell,qp,level)*E.delta(level));
	  const int level_m = level             ? level-1 : 0;
	  const int level_p = level+1<numLevels ? level+1 : level;
	  const ScalarT etadotpi_m = etadotpi[level  ];
	  const ScalarT etadotpi_p = etadotpi[level+1];

	  const ScalarT dT_m       = Temperature(cell,qp,level)   - Temperature(cell,qp,level_m);
	  const ScalarT dT_p       = Temperature(cell,qp,level_p) - Temperature(cell,qp,level);
	  etadotdT(cell,qp,level) = factor * ( etadotpi_p*dT_p + etadotpi_m*dT_m );

   	  for (int dim=0; dim<numDims; ++dim) {
 	    const ScalarT dVx_m      = Velocity(cell,qp,level,dim)   - Velocity(cell,qp,level_m,dim);
	    const ScalarT dVx_p      = Velocity(cell,qp,level_p,dim) - Velocity(cell,qp,level,dim);
	    etadotdVelx(cell,qp,level,dim) = factor * ( etadotpi_p*dVx_p + etadotpi_m*dVx_m );
	  }
	  for (int i = 0; i < tracerNames.size(); ++i) {
	    const ScalarT q_m = 0.5*( Tracer[tracerNames[i]](cell,qp,level)   / Pi(cell,qp,level)
	 		      + Tracer[tracerNames[i]](cell,qp,level_m) / Pi(cell,qp,level_m) );
	    const ScalarT q_p = 0.5*( Tracer[tracerNames[i]](cell,qp,level_p) / Pi(cell,qp,level_p)
	 		      + Tracer[tracerNames[i]](cell,qp,level)   / Pi(cell,qp,level)   );
	    dedotpiTracerde[tracerNames[i]](cell,qp,level) = ( etadotpi_p*q_p - etadotpi_m*q_m ) / E.delta(level);
	  }
  	  Pidot(cell,qp,level) = - divpivelx(cell,qp,level) - (etadotpi_p - etadotpi_m)/E.delta(level);
        }

      }
    }
  }

#else
  // Obtain vector of device views from map of MDFields
  for (int i = 0; i < tracerNames.size(); ++i) {
    Tracer_kokkos[i] = Tracer[tracerNames[i]].get_static_view();
    //etadotdTracer_kokkos[i] = etadotdTracer[tracerNames[i]].get_static_view();
    dedotpiTracerde_kokkos[i] = dedotpiTracerde[tracerNames[i]].get_static_view();
  }
  d_Tracer = Tracer_kokkos.template view<executionSpace>();
  //d_etadotdTracer = etadotdTracer_kokkos.template view<executionSpace>();
  d_dedotpiTracerde = dedotpiTracerde_kokkos.template view<executionSpace>();

  if (!pureAdvection) {
    Kokkos::parallel_for(XZHydrostatic_EtaDotPi_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }

  else {
    Kokkos::parallel_for(XZHydrostatic_EtaDotPi_pureAdvection_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }

#endif
}
}
