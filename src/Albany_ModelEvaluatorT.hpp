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

//9/12/14: no Epetra! 

#ifndef ALBANY_MODELEVALUATORT_HPP
#define ALBANY_MODELEVALUATORT_HPP

#include "Thyra_ModelEvaluatorDefaultBase.hpp"

#include "Albany_Application.hpp"

#include "Teuchos_TimeMonitor.hpp"

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
  Teuchos::ArrayView<const std::string> get_g_names(int j) const
  { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "not impl'ed"); }

  Thyra::ModelEvaluatorBase::InArgs<ST> getNominalValues() const;

  Thyra::ModelEvaluatorBase::InArgs<ST> getLowerBounds() const;

  Thyra::ModelEvaluatorBase::InArgs<ST> getUpperBounds() const;


  Teuchos::RCP<Thyra::LinearOpBase<ST> > create_W_op() const;

  //! Create preconditioner operator
  Teuchos::RCP<Thyra::PreconditionerBase<ST> > create_W_prec() const;

  Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST> > get_W_factory() const;


  //! Create InArgs
  Thyra::ModelEvaluatorBase::InArgs<ST> createInArgs() const;


  void reportFinalPoint(const Thyra::ModelEvaluatorBase::InArgs<ST>& finalPoint, const bool wasSolved);

  void allocateVectors();

  //@}

protected:
  /** \name Overridden from Thyra::ModelEvaluatorDefaultBase<ST> . */
  //@{
   
  //! Create operator form of df/dp for distributed parameters
   Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DfDp_op_impl(int j) const;

  //! Create operator form of dg/dx for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DgDx_op_impl(int j) const;

  //! Create operator form of dg/dx_dot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DgDx_dot_op_impl(int j) const;

  //! Create OutArgs
  Thyra::ModelEvaluatorBase::OutArgs<ST> createOutArgsImpl() const;

  //! Evaluate model on InArgs
  void evalModelImpl(
      const Thyra::ModelEvaluatorBase::InArgs<ST>& inArgs,
      const Thyra::ModelEvaluatorBase::OutArgs<ST>& outArgs) const;

  //! Application object
  Teuchos::RCP<Albany::Application> app;

  Teuchos::RCP<Teuchos::Time> timer;

  //! Sacado parameter vector
  mutable Teuchos::Array<ParamVec> sacado_param_vec;

  //@}

private:

  //! Number of parameter vectors
  int num_param_vecs;

  Thyra::ModelEvaluatorBase::InArgs<ST> createInArgsImpl() const;

  //! Cached nominal values
  Thyra::ModelEvaluatorBase::InArgs<ST> nominalValues;

  //! List of free parameter names
  Teuchos::Array<Teuchos::RCP<Teuchos::Array<std::string> > > param_names;


  //! Tpetra map for parameter vector
  Teuchos::Array<Teuchos::RCP<Tpetra_Map> > tpetra_param_map;

  //! Tpetra parameter vector
  Teuchos::Array<Teuchos::RCP<Tpetra_Vector> > tpetra_param_vec;

  //! Tpetra response vector
  Teuchos::Array<Teuchos::RCP<Thyra::VectorBase<ST> > > thyra_response_vec;

  
  //! Number of distributed parameter vectors
  int num_dist_param_vecs;

  //! List of enabled distributed parameters
  Teuchos::Array<std::string> dist_param_names;

  //! Distributed parameter library
  Teuchos::RCP<DistParamLib> distParamLib;
};

}

#endif // ALBANY_MODELEVALUATORT_HPP
