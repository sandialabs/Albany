//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_FACTORY_TRAITS_HPP
#define PHAL_FACTORY_TRAITS_HPP

// Pull in all Albany configuration macros
#include "Albany_config.h"

// User Defined Evaluator Types

#include "PHAL_SDirichlet.hpp"
#include "PHAL_Dirichlet.hpp"
#include "PHAL_TimeDepDBC.hpp"
#include "PHAL_TimeDepSDBC.hpp"
#include "PHAL_ExprEvalSDBC.hpp"
#include "PHAL_DirichletCoordinateFunction.hpp"
#include "PHAL_DirichletField.hpp"
#include "PHAL_SDirichletField.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_GatherSolution.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_Neumann.hpp"

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
    static const int id_timedep_bc                     =  4; 
    static const int id_timedep_sdbc                   =  5; 
    static const int id_sdbc                           =  6;
    static const int id_sdirichlet_field               =  7;
    static int const id_expreval_sdbc                  =  8; // Only if ALBANY_STK_EXPR_EVAL is ON

    typedef Sacado::mpl::vector<
        PHAL::Dirichlet<_,Traits>,                //  0
        PHAL::DirichletAggregator<_,Traits>,      //  1
        PHAL::DirichletCoordFunction<_,Traits>,   //  2
        PHAL::DirichletField<_,Traits>,           //  3
        PHAL::TimeDepDBC<_, Traits>,              //  4
        PHAL::TimeDepSDBC<_, Traits>,             //  5
        PHAL::SDirichlet<_, Traits>,              //  6
        PHAL::SDirichletField<_, Traits>          //  7
#ifdef ALBANY_STK_EXPR_EVAL
        ,
        PHAL::ExprEvalSDBC<_, Traits>            //   9
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


    typedef Sacado::mpl::vector<
       PHAL::Neumann<_,Traits>,                   //  0
       PHAL::NeumannAggregator<_,Traits>,         //  1
       PHAL::GatherCoordinateVector<_,Traits>,    //  2
       PHAL::GatherSolution<_,Traits>,            //  3
       PHAL::LoadStateField<_,Traits>,            //  4
       PHAL::GatherScalarNodalParameter<_,Traits> //  5
    > EvaluatorTypes;
};

}

#endif
