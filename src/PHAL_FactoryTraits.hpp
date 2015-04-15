//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef PHAL_FACTORY_TRAITS_HPP
#define PHAL_FACTORY_TRAITS_HPP

// User Defined Evaluator Types

#if defined(ALBANY_LCM)
#include "LCM/evaluators/bc/KfieldBC.hpp"
#include "LCM/evaluators/bc/TimeDepBC.hpp"
#include "LCM/evaluators/bc/TimeTracBC.hpp"
#include "LCM/evaluators/Time.hpp"
#include "LCM/evaluators/bc/SchwarzBC.hpp"
#include "LCM/evaluators/bc/TorsionBC.hpp"
#endif
#ifdef ALBANY_QCAD
#include "QCAD_PoissonDirichlet.hpp"
#include "QCAD_PoissonNeumann.hpp"
#endif
#include "PHAL_Dirichlet.hpp"
#include "PHAL_Neumann.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_GatherSolution.hpp"
#ifdef ALBANY_EPETRA
#include "PHAL_GatherAuxData.hpp"
#endif
#include "PHAL_LoadStateField.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_DirichletCoordinateFunction.hpp"
#include "PHAL_DirichletField.hpp"


#include "boost/mpl/vector/vector50.hpp"
#include "boost/mpl/placeholders.hpp"

// \cond  Have doxygern ignore this namespace
using namespace boost::mpl::placeholders;
// \endcond

namespace PHAL {
/*! \brief Struct to define Evaluator objects for the EvaluatorFactory.

    Preconditions:
    - You must provide a boost::mpl::vector named EvaluatorTypes that contain
    all Evaluator objects that you wish the factory to build.  Do not confuse
    evaluator types (concrete instances of evaluator objects) with evaluation
    types (types of evaluations to perform, i.e., Residual, Jacobian).

*/

  template<typename Traits>
  struct DirichletFactoryTraits {

    static const int id_dirichlet                      =  0;
    static const int id_dirichlet_aggregator           =  1;
    static const int id_dirichlet_coordinate_function  =  2;
    static const int id_dirichlet_field                =  3;
    static const int id_qcad_poisson_dirichlet         =  4;
    static const int id_kfield_bc                      =  5; // Only for LCM probs
    static const int id_timedep_bc                     =  6; // Only for LCM probs
    static const int id_time                           =  7; // Only for LCM probs
    static const int id_torsion_bc                     =  8; // Only for LCM probs
    static const int id_schwarz_bc                     =  9; // Only for LCM probs

#if defined(ALBANY_LCM)
    typedef boost::mpl::vector10<
#else
      typedef boost::mpl::vector5<
#endif
        PHAL::Dirichlet<_,Traits>,                //  0
        PHAL::DirichletAggregator<_,Traits>,      //  1
        PHAL::DirichletCoordFunction<_,Traits>,   //  2
        PHAL::DirichletField<_,Traits>,           //  3
#ifdef ALBANY_QCAD
        QCAD::PoissonDirichlet<_,Traits>          //  4
#else
        PHAL::Dirichlet<_,Traits>                 //  4 dummy
#endif
#if defined(ALBANY_LCM)
        ,
        LCM::KfieldBC<_,Traits>,                  //  5
        LCM::TimeDepBC<_, Traits>,                //  6
        LCM::Time<_, Traits>,                     //  7
        LCM::TorsionBC<_, Traits>,                //  8
        LCM::SchwarzBC<_, Traits>                 //  9
#endif
        > EvaluatorTypes;
};


  template<typename Traits>
  struct NeumannFactoryTraits {

    static const int id_neumann                   =  0;
    static const int id_neumann_aggregator        =  1;
    static const int id_gather_coord_vector       =  2;
    static const int id_gather_solution           =  3;
    static const int id_load_stateField           =  4;
    static const int id_GatherScalarNodalParameter=  5;
    static const int id_qcad_poisson_neumann      =  6; // Only for QCAD probs
    static const int id_timedep_bc                =  7; // Only for LCM probs


#if defined(ALBANY_LCM)
    typedef boost::mpl::vector8<
#else
    typedef boost::mpl::vector7<
#endif
	     PHAL::Neumann<_,Traits>,                   //  0
	     PHAL::NeumannAggregator<_,Traits>,         //  1
       PHAL::GatherCoordinateVector<_,Traits>,    //  2
       PHAL::GatherSolution<_,Traits>,            //  3
       PHAL::LoadStateField<_,Traits>,            //  4
       PHAL::GatherScalarNodalParameter<_,Traits>,//  5
#ifdef ALBANY_QCAD
       QCAD::PoissonNeumann<_,Traits>             //  6
#else
       PHAL::Neumann<_,Traits>                    //  6 dummy
#endif
#if defined(ALBANY_LCM)
       , LCM::TimeTracBC<_, Traits>               //  7
#endif
	  > EvaluatorTypes;
};

}

#endif
