//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_COUPLEDPOISSONSCHRODINGER_H
#define QCAD_COUPLEDPOISSONSCHRODINGER_H

#include <iostream>

#include "LOCA.H"
#include "LOCA_Epetra.H"
#include "Epetra_Vector.h"
#include "Epetra_LocalMap.h"
#include "LOCA_Epetra_ModelEvaluatorInterface.H"
#include <NOX_Epetra_MultiVector.H>

#include "Albany_ModelEvaluator.hpp"
#include "Albany_Utils.hpp"
#include "Piro_Epetra_StokhosNOXObserver.hpp"

namespace QCAD {

/** \brief Epetra-based Model Evaluator for QCAD coupled poisson-schrodinger solver
 *
 */

  class CoupledPoissonSchrodinger : public EpetraExt::ModelEvaluator {
  public:

    /** \name Constructors/initializers */
    //@{

      CoupledPoissonSchrodinger(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
				const Teuchos::RCP<const Epetra_Comm>& comm, 
				const Teuchos::RCP<const Epetra_Vector>& initial_guess);
    //@}

    ~CoupledPoissonSchrodinger();

    Teuchos::RCP<const Epetra_Map> get_x_map() const;
    Teuchos::RCP<const Epetra_Map> get_f_map() const;
    Teuchos::RCP<const Epetra_Map> get_p_map(int l) const;
    Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

    Teuchos::RCP<const Teuchos::Array<std::string> > get_p_names(int l) const;

    Teuchos::RCP<const Epetra_Vector> get_x_init() const;
    Teuchos::RCP<const Epetra_Vector> get_x_dot_init() const;
    Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;

    Teuchos::RCP<Epetra_Operator> create_W() const;
    Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner> create_WPrec() const;

    Teuchos::RCP<Epetra_Operator> create_DgDx_op(int j) const;
    Teuchos::RCP<Epetra_Operator> create_DgDx_dot_op(int j) const;

    EpetraExt::ModelEvaluator::InArgs createInArgs() const;
    EpetraExt::ModelEvaluator::OutArgs createOutArgs() const;

    void evalModel( const InArgs& inArgs, const OutArgs& outArgs ) const;    


    Teuchos::RCP<Albany::Application> getPoissonApp() const; //for creating NOX observers (temporary)

  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidAppParameters() const;


  private:
    Teuchos::RCP<const Epetra_Map> disc_map, disc_overlap_map;
    Teuchos::RCP<Epetra_Map> combined_SP_map;

    Teuchos::RCP<Albany::Application> poissonApp, schrodingerApp;
    Teuchos::RCP<EpetraExt::ModelEvaluator> poissonModel, schrodingerModel;

    Teuchos::RCP<const Epetra_Comm> myComm;

    int nEigenvals;
    int num_param_vecs, num_response_vecs;
    int num_poisson_param_vecs, num_schrodinger_param_vecs;

    //! Sacado parameter vectors
    mutable Teuchos::Array<ParamVec> poisson_sacado_param_vec, schrodinger_sacado_param_vec;

    bool bVerbose;
  };
}
#endif
