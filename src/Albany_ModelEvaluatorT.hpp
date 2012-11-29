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


#ifndef ALBANY_MODELEVALUATORT_HPP
#define ALBANY_MODELEVALUATORT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include "Albany_Application.hpp"
#include "EpetraExt_ModelEvaluator.h"
#include "Epetra_LocalMap.h"

//Thyra includes 
#include "Thyra_ModelEvaluatorDefaultBase.hpp"

#include "Thyra_LinearOpWithSolveBase.hpp"
#include "Thyra_LinearOpWithSolveFactoryBase.hpp"

using namespace Thyra; 

namespace Albany {

  class ModelEvaluatorT : public Thyra::ModelEvaluatorDefaultBase<ST> {
  public:

    // Constructor
    ModelEvaluatorT(
       const Teuchos::RCP<Albany::Application>& app,
       const Teuchos::RCP<Teuchos::ParameterList>& appParams);

    /** \name Overridden from Thyra::ModelEvaluator<ST> . */
    //@{

    //! Return solution vector map
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_x_space() const;

    //! Return residual vector map
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_f_space() const;

    //! Return parameter vector map
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_p_space(int l) const;

    //! Return response function map
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST> > get_g_space(int j) const;

    //! Return array of parameter names
    Teuchos::RCP<const Teuchos::Array<std::string> > 
    get_p_names(int l) const;

    Thyra::ModelEvaluatorBase::InArgs<ST> getNominalValues() const; 
    
    Teuchos::RCP<LinearOpBase<ST> >create_W_op() const; 

    //*********CONCRETE VERSIONS OF THE FOLLOWING NEED TO BE OVERWRITTEN FROM THYRA::MODELEVALUATORDEFAULTBASE<ST>.  
   // int Np() const {};  
   // int Ng() const {};  
    Thyra::ModelEvaluatorBase::InArgs<ST> getLowerBounds() const {}; 
    Thyra::ModelEvaluatorBase::InArgs<ST> getUpperBounds() const {}; 
    //Teuchos::RCP<LinearOpWithSolveBase<ST> >create_W() const {}; 
    //Teuchos::RCP<LinearOpBase<ST> >create_DfDp_op(int j) const {}; 
    //Teuchos::RCP<LinearOpBase<ST> >create_DgDp_op(int j, int l) const {}; 
    Teuchos::RCP<const LinearOpWithSolveFactoryBase<ST> >get_W_factory() const {}; 
    void reportFinalPoint(const Thyra::ModelEvaluatorBase::InArgs<ST>& finalPoint, const bool wasSolved) {}; 
   
   //**************************  
 
    //! Create preconditioner operator
    Teuchos::RCP<Thyra::PreconditionerBase<ST> > create_W_prec() const;

    //! Create operator form of dg/dx for distributed responses
    Teuchos::RCP<LinearOpBase<ST> > create_DgDx_op(int j) const;

    //! Create operator form of dg/dx_dot for distributed responses
    Teuchos::RCP<LinearOpBase<ST> > create_DgDx_dot_op(int j) const;
    
    //! Create InArgs
    Thyra::ModelEvaluatorBase::InArgs<ST> createInArgs() const;

    //! Create OutArgs
    Thyra::ModelEvaluatorBase::OutArgs<ST> createOutArgsImpl() const; 

    //! Evaluate model on InArgs
    void evalModelImpl(const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgs, const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgs) const; 

    //@}

  protected:

    //! Application object
    Teuchos::RCP<Albany::Application> app;

    Thyra::ModelEvaluatorBase::InArgs<ST> nominalValues; 
    Thyra::ModelEvaluatorBase::InArgs<ST> prototypeInArgsT; 
    Thyra::ModelEvaluatorBase::OutArgs<ST> prototypeOutArgsT; 

    //! List of free parameter names
    Teuchos::Array< Teuchos::RCP< Teuchos::Array<std::string> > > param_names;

    //! Sacado parameter vector
    mutable Teuchos::Array<ParamVec> sacado_param_vec;

    //! Tpetra map for parameter vector
    Teuchos::Array< Teuchos::RCP<Tpetra_Map> > tpetra_param_map;

    //! Tpetra parameter vector
    Teuchos::Array< Teuchos::RCP<Tpetra_Vector> > tpetra_param_vec;

    //! Whether the problem supplies its own preconditioner
    bool supplies_prec;

    //! Stochastic Galerkin parameters
    mutable Teuchos::Array< Teuchos::Array<SGType> > p_sg_vals;

    //! Multi-point parameters
    mutable Teuchos::Array< Teuchos::Array<MPType> > p_mp_vals;

    //! Allocated Jacobian for sending to user preconditioner
    //mutable Teuchos::RCP<Epetra_CrsMatrix> Extra_W_crs;

    Teuchos::RCP<Teuchos::Time> timer;
  };

}

#endif // ALBANY_MODELEVALUATORT_HPP
