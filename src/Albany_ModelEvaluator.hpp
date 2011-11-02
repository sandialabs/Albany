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

    //! Return initial parameters
    Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;

    //! Create W = alpha*M + beta*J matrix
    Teuchos::RCP<Epetra_Operator> create_W() const;

    //! Create preconditioner operator
    Teuchos::RCP<EpetraExt::ModelEvaluator::Preconditioner> create_WPrec() const;

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
  };

}

#endif // ALBANY_MODELEVALUATOR_HPP
