//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_GENEIGENSOLVER_H
#define QCAD_GENEIGENSOLVER_H

#include <iostream>

//#include "LOCA.H"
//#include "LOCA_Epetra.H"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
//#include "Epetra_LocalMap.h"
#include "EpetraExt_ModelEvaluator.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_StateManager.hpp"

//#include "LOCA_Epetra_ModelEvaluatorInterface.H"
//#include <NOX_Epetra_MultiVector.H>

//#include "Albany_ModelEvaluator.hpp"
//#include "Albany_Utils.hpp"
//#include "Piro_Epetra_StokhosNOXObserver.hpp"


namespace QCAD {

/** \brief Epetra-based Model Evaluator for QCAD Generalized Eigensolver
 *
 */

  class GenEigensolver : public EpetraExt::ModelEvaluator {
  public:

    /** \name Constructors/initializers */
    //@{

      GenEigensolver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
		     const Teuchos::RCP<EpetraExt::ModelEvaluator>& model,
		     const Teuchos::RCP<Albany::StateManager>& observer,
		     Teuchos::RCP<const Epetra_Comm> comm);
    //@}

    ~GenEigensolver();

    Teuchos::RCP<const Epetra_Map> get_x_map() const;
    Teuchos::RCP<const Epetra_Map> get_f_map() const;
    Teuchos::RCP<const Epetra_Map> get_p_map(int l) const;
    Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

    Teuchos::RCP<const Epetra_Vector> get_x_init() const;
    Teuchos::RCP<const Epetra_Vector> get_x_dot_init() const;
    Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;

    EpetraExt::ModelEvaluator::InArgs createInArgs() const;
    EpetraExt::ModelEvaluator::OutArgs createOutArgs() const;

    void evalModel( const InArgs& inArgs, const OutArgs& outArgs ) const;    

  private:
    Teuchos::RCP<EpetraExt::ModelEvaluator> model;
    Teuchos::RCP<Albany::StateManager> observer; //use a state manager as an observer (holds eigen data)
    int model_num_p, model_num_g;

    Teuchos::RCP<const Epetra_Comm> myComm;

    //Eigensolver parameters
    bool bHermitian;
    std::string which;
    int nev, blockSize, maxIters;
    double conv_tol;
  };
}
#endif
