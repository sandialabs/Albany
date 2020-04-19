//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef PHAL_FACTORY_TRAITS_HPP
#define PHAL_FACTORY_TRAITS_HPP

// Pull in all Albany configuration macros
#include "Albany_config.h"

// User Defined Evaluator Types

#if defined(ALBANY_LCM)
#include "LCM/evaluators/bc/EquilibriumConcentrationBC.hpp"
#include "LCM/evaluators/bc/KfieldBC.hpp"
#include "LCM/evaluators/bc/PDNeighborFitBC.hpp"
#include "LCM/evaluators/bc/TimeTracBC.hpp"
#include "LCM/evaluators/bc/TorsionBC.hpp"
#include "LCM/evaluators/Time.hpp"
#if defined(ALBANY_STK) 
#include "LCM/evaluators/bc/SchwarzBC.hpp"
#include "LCM/evaluators/bc/StrongSchwarzBC.hpp"
#endif // ALBANY_STK
#endif // ALBANY_LCM

#include "PHAL_SDirichlet.hpp"
#include "PHAL_Dirichlet.hpp"
#include "PHAL_TimeDepDBC.hpp"
#include "PHAL_TimeDepSDBC.hpp"
#include "PHAL_ExprEvalSDBC.hpp"
#include "PHAL_DirichletCoordinateFunction.hpp"
#include "PHAL_DirichletField.hpp"
#include "PHAL_SDirichletField.hpp"
#include "PHAL_DirichletOffNodeSet.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_GatherSolution.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_Neumann.hpp"

#if defined(ALBANY_EPETRA)
#include "PHAL_GatherAuxData.hpp"
#endif

#include "Sacado_mpl_placeholders.hpp"

// \cond  Have doxygern ignore this namespace
using namespace Sacado::mpl::placeholders;
// \endcond

namespace PHAL {
/*! \brief Struct to define Evaluator objects for the EvaluatorFactory.

    Preconditions:
    - You must provide a Sacado::mpl::vector named EvaluatorTypes that contain
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
    static const int id_dirichlet_off_nodeset          =  4; // To handle equations on side set (see PHAL_DirichletOffNodeSet)
    static const int id_timedep_bc                     =  5; 
    static const int id_timedep_sdbc                   =  6; 
    static const int id_sdbc                           =  7;
    static const int id_sdirichlet_field               =  8;
    static const int id_kfield_bc                      =  9; // Only for LCM probs
    static const int id_eq_concentration_bc            = 10; // Only for LCM probs
    static const int id_time                           = 11; // Only for LCM probs
    static const int id_torsion_bc                     = 12; // Only for LCM probs
    static const int id_schwarz_bc                     = 13; // Only for LCM probs
    static const int id_strong_schwarz_bc              = 14; // Only for LCM probs
    static int const id_expreval_sdbc                  = 15; // Only if ALBANY_STK_EXPR_EVAL is ON

    typedef Sacado::mpl::vector<
        PHAL::Dirichlet<_,Traits>,                //  0
        PHAL::DirichletAggregator<_,Traits>,      //  1
        PHAL::DirichletCoordFunction<_,Traits>,   //  2
        PHAL::DirichletField<_,Traits>,           //  3
        PHAL::DirichletOffNodeSet<_,Traits>,      //  4
        PHAL::TimeDepDBC<_, Traits>,              //  5
        PHAL::TimeDepSDBC<_, Traits>,             //  6
        PHAL::SDirichlet<_, Traits>,              //  7
        PHAL::SDirichletField<_, Traits>          //  8
#if defined(ALBANY_LCM)
        ,
        LCM::KfieldBC<_,Traits>,                   // 9
        LCM::EquilibriumConcentrationBC<_,Traits>, // 10
        LCM::Time<_, Traits>,                      // 11
        LCM::TorsionBC<_, Traits>                  // 12
#endif
#if defined(ALBANY_LCM) && defined(ALBANY_STK) 
        ,
        LCM::SchwarzBC<_, Traits>,                 // 13
        LCM::StrongSchwarzBC<_, Traits>,           // 14
        LCM::PDNeighborFitBC<_, Traits>            // 15
#endif
#ifdef ALBANY_STK_EXPR_EVAL
        ,
        PHAL::ExprEvalSDBC<_, Traits>            //  16
#endif
        > EvaluatorTypes;
};


  template<typename Traits>
  struct NeumannFactoryTraits {

    static const int id_neumann                    =  0;
    static const int id_neumann_aggregator         =  1;
    static const int id_gather_coord_vector        =  2;
    static const int id_gather_solution            =  3;
    static const int id_load_stateField            =  4;
    static const int id_GatherScalarNodalParameter =  5;
    static const int id_timedep_bc                 =  6; // Only for LCM probs


    typedef Sacado::mpl::vector<
       PHAL::Neumann<_,Traits>,                   //  0
       PHAL::NeumannAggregator<_,Traits>,         //  1
       PHAL::GatherCoordinateVector<_,Traits>,    //  2
       PHAL::GatherSolution<_,Traits>,            //  3
       PHAL::LoadStateField<_,Traits>,            //  4
       PHAL::GatherScalarNodalParameter<_,Traits> //  5
#if defined(ALBANY_LCM)
       , LCM::TimeTracBC<_, Traits>               //  6
#endif
    > EvaluatorTypes;
};

}

#endif
