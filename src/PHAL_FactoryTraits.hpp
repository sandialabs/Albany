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


#ifndef PHAL_FACTORY_TRAITS_HPP
#define PHAL_FACTORY_TRAITS_HPP

// User Defined Evaluator Types
#include "PHAL_Constant.hpp"
#include "PHAL_Dirichlet.hpp"
#include "PHAL_GatherSolution.hpp"
#include "PHAL_ScatterResidual.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_ThermalConductivity.hpp"
#include "PHAL_ComputeBasisFunctions.hpp"
#include "PHAL_DOFInterpolation.hpp"
#include "PHAL_DOFGradInterpolation.hpp"
#include "PHAL_DOFVecInterpolation.hpp"
#include "PHAL_DOFVecGradInterpolation.hpp"
#include "PHAL_MapToPhysicalFrame.hpp"
#include "PHAL_HelmholtzResid.hpp"
#include "PHAL_HeatEqResid.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "QCAD_Permittivity.hpp"
#include "QCAD_PoissonResid.hpp"
#include "QCAD_PoissonSource.hpp"
#include "PHAL_JouleHeating.hpp"
#include "PHAL_TEProp.hpp"
#include "PHAL_ODEResid.hpp"

#include "boost/mpl/vector/vector30.hpp"
#include "boost/mpl/placeholders.hpp"
using namespace boost::mpl::placeholders;

/*! \brief Struct to define Evaluator objects for the EvaluatorFactory.
    
    Preconditions:
    - You must provide a boost::mpl::vector named EvaluatorTypes that contain all Evaluator objects that you wish the factory to build.  Do not confuse evaluator types (concrete instances of evaluator objects) with evaluation types (types of evaluations to perform, i.e., Residual, Jacobian). 

*/
namespace PHAL {

template<typename Traits>
struct FactoryTraits {
  
  static const int id_dirichlet                 =  0;
  static const int id_gather_solution           =  1;
  static const int id_gather_coordinate_vector  =  2;
  static const int id_scatter_residual          =  3;
  static const int id_compute_basis_functions   =  4;
  static const int id_dof_interpolation         =  5;
  static const int id_dof_grad_interpolation    =  6;
  static const int id_dofvec_interpolation      =  7;
  static const int id_dofvec_grad_interpolation =  8;
  static const int id_map_to_physical_frame     =  9;
  static const int id_source                    = 10;
  static const int id_thermal_conductivity      = 11;
  static const int id_helmholtzresid            = 12;
  static const int id_heateqresid               = 13;
  static const int id_constant                  = 14;
  static const int id_dirichlet_aggregator      = 15;
  static const int id_qcad_permittivity         = 16;
  static const int id_qcad_poisson_resid        = 17;
  static const int id_qcad_poisson_source       = 18;
  static const int id_jouleheating              = 19;
  static const int id_teprop                    = 20;
  static const int id_oderesid                  = 21;

  typedef boost::mpl::vector22< 
            PHAL::Dirichlet<_,Traits>,                //  0
            PHAL::GatherSolution<_,Traits>,           //  1
            PHAL::GatherCoordinateVector<_,Traits>,   //  2
            PHAL::ScatterResidual<_,Traits>,          //  3
            PHAL::ComputeBasisFunctions<_,Traits>,    //  4
            PHAL::DOFInterpolation<_,Traits>,         //  5
            PHAL::DOFGradInterpolation<_,Traits>,     //  6
            PHAL::DOFVecInterpolation<_,Traits>,      //  7
            PHAL::DOFVecGradInterpolation<_,Traits>,  //  8
            PHAL::MapToPhysicalFrame<_,Traits>,       //  9
            PHAL::Source<_,Traits>,                   // 10
            PHAL::ThermalConductivity<_,Traits>,      // 11
            PHAL::HelmholtzResid<_,Traits>,           // 12
            PHAL::HeatEqResid<_,Traits>,              // 13
            PHAL::Constant<_,Traits>,                 // 14
            PHAL::DirichletAggregator<_,Traits>,      // 15
            QCAD::Permittivity<_,Traits>,             // 16
            QCAD::PoissonResid<_,Traits>,             // 17
            QCAD::PoissonSource<_,Traits>,            // 18
            PHAL::JouleHeating<_,Traits>,             // 19
            PHAL::TEProp<_,Traits>,                   // 20
            PHAL::ODEResid<_,Traits>                  // 21
  > EvaluatorTypes;
  
};
}

#endif

