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

#ifdef ALBANY_LCM
#include "LCM/evaluators/KfieldBC.hpp"
#include "LCM/evaluators/TimeDepBC.hpp"
#include "LCM/evaluators/Time.hpp"
#include "LCM/evaluators/TorsionBC.hpp"
#endif
#include "QCAD_PoissonDirichlet.hpp"
#include "PHAL_Dirichlet.hpp"
#include "PHAL_Neumann.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_GatherSolution.hpp"


#include "boost/mpl/vector/vector50.hpp"
#include "boost/mpl/placeholders.hpp"

// \cond  Have doxygern ignore this namespace 
using namespace boost::mpl::placeholders;
// \endcond

namespace PHAL {
/*! \brief Struct to define Evaluator objects for the EvaluatorFactory.

    Preconditions:
    - You must provide a boost::mpl::vector named EvaluatorTypes that contain all Evaluator objects that you wish the factory to build.  Do not confuse evaluator types (concrete instances of evaluator objects) with evaluation types (types of evaluations to perform, i.e., Residual, Jacobian). 

*/

  template<typename Traits>
  struct DirichletFactoryTraits {

    static const int id_dirichlet                 =  0;
    static const int id_dirichlet_aggregator      =  1;
    static const int id_qcad_poisson_dirichlet    =  2;
    static const int id_kfield_bc                 =  3; // Only for LCM probs
    static const int id_timedep_bc                =  4; // Only for LCM probs
    static const int id_time                      =  5; // Only for LCM probs
    static const int id_torsion_bc                =  6; // Only for LCM probs

#ifdef ALBANY_LCM
    typedef boost::mpl::vector7<
#else
      typedef boost::mpl::vector3<
#endif
        PHAL::Dirichlet<_,Traits>,                //  0
        PHAL::DirichletAggregator<_,Traits>,      //  1
        QCAD::PoissonDirichlet<_,Traits>          //  2
#ifdef ALBANY_LCM
        , LCM::KfieldBC<_,Traits>,                //  3
        LCM::TimeDepBC<_, Traits>,                //  4
        LCM::Time<_, Traits>,                     //  5
        LCM::TorsionBC<_, Traits>                 //  6
#endif
        > EvaluatorTypes;
};


  template<typename Traits>
  struct NeumannFactoryTraits {
  
    static const int id_neumann                   =  0;
    static const int id_neumann_aggregator        =  1;
    static const int id_gather_coord_vector       =  2;
    static const int id_gather_solution           =  3;
    static const int id_timedep_bc                =  4; // Only for LCM probs

#ifdef ALBANY_LCM
    typedef boost::mpl::vector5<
#else
    typedef boost::mpl::vector4< 
#endif

	     PHAL::Neumann<_,Traits>,                     //  0
	     PHAL::NeumannAggregator<_,Traits>,           //  1
             PHAL::GatherCoordinateVector<_,Traits>,      //  2
             PHAL::GatherSolution<_,Traits>               //  3
#ifdef ALBANY_LCM
        , LCM::TimeDepBC<_, Traits>                //  4
#endif

	  > EvaluatorTypes;
};

}

#endif
