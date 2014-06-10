//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MODELEVALUATOR_HPP
#define ALBANY_MODELEVALUATOR_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include "Albany_Application.hpp"
#include "EpetraExt_ModelEvaluator.h"
#include "Epetra_LocalMap.h"

namespace Albany {

  class ModelEvaluator : public EpetraExt::ModelEvaluator {
  public:

    // Constructor
    ModelEvaluator(
       const Teuchos::RCP<Albany::Application>& app,
       const Teuchos::RCP<Teuchos::ParameterList>& appParams);

    virtual ~ModelEvaluator();

    /** \name Overridden from EpetraExt::ModelEvaluator . */
    //@{

    //! Return solution vector map
    Teuchos::RCP<const Epetra_Map> get_x_map() const;

    //! Return residual vector map
    Teuchos::RCP<const Epetra_Map> get_f_map() const;

    //! Return parameter vector map
    Teuchos::RCP<const Epetra_Map> get_p_map(int l) const;

    //! Return response function map
    Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

    //! Return array of parameter names
    Teuchos::RCP<const Teuchos::Array<std::string> >
    get_p_names(int l) const;

    //! Return initial solution and x_dot init
    Teuchos::RCP<const Epetra_Vector> get_x_init() const;
    Teuchos::RCP<const Epetra_Vector> get_x_dot_init() const;
    Teuchos::RCP<const Epetra_Vector> get_x_dotdot_init() const;

    //! Return initial parameters
    Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;

    //! Create W = alpha*M + beta*J + omega*N matrix
    Teuchos::RCP<Epetra_Operator> create_W() const;

    //! Create preconditioner operator
    Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner> create_WPrec() const;

    //! Create operator form of df/dp for distributed parameters
    Teuchos::RCP<Epetra_Operator> create_DfDp_op(int j) const;

    //! Create operator form of dg/dx for distributed responses
    Teuchos::RCP<Epetra_Operator> create_DgDx_op(int j) const;

    //! Create operator form of dg/dx_dot for distributed responses
    Teuchos::RCP<Epetra_Operator> create_DgDx_dot_op(int j) const;
    Teuchos::RCP<Epetra_Operator> create_DgDx_dotdot_op(int j) const;

    //! Create InArgs
    InArgs createInArgs() const;

    //! Create OutArgs
    OutArgs createOutArgs() const;

    //! Evaluate model on InArgs
    void evalModel(const InArgs& inArgs, const OutArgs& outArgs) const;

    //@}

  protected:

    //! Application object
    Teuchos::RCP<Albany::Application> app;

    //! Number of parameter vectors
    int num_param_vecs;

    //! List of free parameter names
    Teuchos::Array< Teuchos::RCP< Teuchos::Array<std::string> > > param_names;

    //! Sacado parameter vector
    mutable Teuchos::Array<ParamVec> sacado_param_vec;

    //! Epetra map for parameter vector
    Teuchos::Array< Teuchos::RCP<Epetra_LocalMap> > epetra_param_map;

    //! Epetra parameter vector
    Teuchos::Array< Teuchos::RCP<Epetra_Vector> > epetra_param_vec;

    //! Whether the problem supplies its own preconditioner
    bool supplies_prec;

    //! Stochastic Galerkin parameters
    mutable Teuchos::Array< Teuchos::Array<SGType> > p_sg_vals;

    //! Multi-point parameters
    mutable Teuchos::Array< Teuchos::Array<MPType> > p_mp_vals;

    //! Allocated Jacobian for sending to user preconditioner
    mutable Teuchos::RCP<Epetra_CrsMatrix> Extra_W_crs;

    Teuchos::RCP<Teuchos::Time> timer;

    //! Number of distributed parameter vectors
    int num_dist_param_vecs;

    //! List of enabled distributed parameters
    Teuchos::Array<std::string> dist_param_names;

    //! Distributed parameter library
    Teuchos::RCP<DistParamLib> distParamLib;
  };

}

#endif // ALBANY_MODELEVALUATOR_HPP
