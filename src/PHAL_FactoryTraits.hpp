//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
#include "LCM/evaluators/bc/EquilibriumConcentrationBC.hpp"
#include "LCM/evaluators/Time.hpp"
#if defined(HAVE_STK)
#include "LCM/evaluators/bc/SchwarzBC.hpp"
#endif
#include "LCM/evaluators/bc/TorsionBC.hpp"
#include "LCM/evaluators/bc/PDNeighborFitBC.hpp"
#endif
#ifdef ALBANY_QCAD
#include "QCAD_PoissonDirichlet.hpp"
#include "QCAD_PoissonNeumann.hpp"
#include "QCAD_PoissonSourceNeumann.hpp"
#include "QCAD_PoissonSourceInterface.hpp"
#endif
#include "PHAL_Dirichlet.hpp"
#include "PHAL_Neumann.hpp"
#include "PHAL_GatherCoordinateVector.hpp"
#include "PHAL_GatherSolution.hpp"
#if defined(ALBANY_EPETRA)
#include "PHAL_GatherAuxData.hpp"
#endif
#include "PHAL_LoadStateField.hpp"
#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_DirichletCoordinateFunction.hpp"
#include "PHAL_DirichletField.hpp"
#include "PHAL_DirichletOffNodeSet.hpp"

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
    static const int id_qcad_poisson_dirichlet         =  5;
    static const int id_kfield_bc                      =  6; // Only for LCM probs
    static const int id_eq_concentration_bc            =  7; // Only for LCM probs
    static const int id_timedep_bc                     =  8; // Only for LCM probs
    static const int id_time                           =  9; // Only for LCM probs
    static const int id_torsion_bc                     = 10; // Only for LCM probs
    static const int id_schwarz_bc                     = 11; // Only for LCM probs
    static const int id_pd_neigh_fit_bc                = 12; // Only for LCM-Peridigm coupling

    typedef Sacado::mpl::vector<
        PHAL::Dirichlet<_,Traits>,                //  0
        PHAL::DirichletAggregator<_,Traits>,      //  1
        PHAL::DirichletCoordFunction<_,Traits>,   //  2
        PHAL::DirichletField<_,Traits>,           //  3
        PHAL::DirichletOffNodeSet<_,Traits>,      //  4
#ifdef ALBANY_QCAD
        QCAD::PoissonDirichlet<_,Traits>          //  5
#else
        PHAL::Dirichlet<_,Traits>                 //  5 dummy
#endif
#if defined(ALBANY_LCM)
        ,
        LCM::KfieldBC<_,Traits>,                  //  6
        LCM::EquilibriumConcentrationBC<_,Traits>, // 7
        LCM::TimeDepBC<_, Traits>,                //  8
        LCM::Time<_, Traits>,                     //  9
        LCM::TorsionBC<_, Traits>                 // 10
#endif
#if defined(ALBANY_LCM) && defined(HAVE_STK)
        ,
        LCM::SchwarzBC<_, Traits>,                 // 11
        LCM::PDNeighborFitBC<_, Traits>           //  12
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
    static const int id_qcad_poisson_neumann       =  6; // Only for QCAD probs
    static const int id_qcad_poissonsource_neumann =  7; // Only for QCAD probs
    static const int id_qcad_poissonsource_interface =  8; // Only for QCAD probs
    static const int id_timedep_bc                =  9; // Only for LCM probs


    typedef Sacado::mpl::vector<
       PHAL::Neumann<_,Traits>,                   //  0
       PHAL::NeumannAggregator<_,Traits>,         //  1
       PHAL::GatherCoordinateVector<_,Traits>,    //  2
       PHAL::GatherSolution<_,Traits>,            //  3
       PHAL::LoadStateField<_,Traits>,            //  4
       PHAL::GatherScalarNodalParameter<_,Traits>,//  5
#ifdef ALBANY_QCAD
       QCAD::PoissonNeumann<_,Traits>,            //  6
       QCAD::PoissonSourceNeumann<_,Traits>,      //  7
       QCAD::PoissonSourceInterface<_,Traits>     //  8
#else
       PHAL::Neumann<_,Traits>,                   //  6 dummy
       PHAL::Neumann<_,Traits>,                   //  7 dummy
       PHAL::Neumann<_,Traits>                    //  8 dummy
#endif
#if defined(ALBANY_LCM)
       , LCM::TimeTracBC<_, Traits>               //  9
#endif
    > EvaluatorTypes;
};

}

#endif
