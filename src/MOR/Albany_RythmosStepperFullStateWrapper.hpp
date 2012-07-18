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

#ifndef ALBANY_RYTHMOSSTEPPERFULLSTATEWRAPPER_HPP
#define ALBANY_RYTHMOSSTEPPERFULLSTATEWRAPPER_HPP

#include "Rythmos_StepperBase.hpp"

class Epetra_Map;

namespace Albany {

class ReducedSpace;

class RythmosStepperFullStateWrapper : public Rythmos::StepperBase<double> {
public:
  RythmosStepperFullStateWrapper(const Teuchos::RCP<const Rythmos::StepperBase<double> > &wrappedStepper,
                                 const Teuchos::RCP<const ReducedSpace> &reducedSpace,
                                 const Teuchos::RCP<const Epetra_Map> &fullMap);

  //
  // Overriden from Teuchos::ParameterListAcceptor
  //

  virtual void setParameterList(const Teuchos::RCP<Teuchos::ParameterList> &paramList);
  virtual Teuchos::RCP<Teuchos::ParameterList> getNonconstParameterList();
  virtual Teuchos::RCP<Teuchos::ParameterList> unsetParameterList();

  //
  // Overriden from Rythmos::InterpolationBufferBase<double>
  //

  virtual Teuchos::RCP<const Thyra::VectorSpaceBase<double> > get_x_space() const;

  virtual void addPoints(const Teuchos::Array<double> &time_vec,
                         const Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > &x_vec,
                         const Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > &xdot_vec);
  virtual void getPoints(const Teuchos::Array<double> &time_vec,
                         Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > *x_vec,
                         Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > *xdot_vec,
                         Teuchos::Array<ScalarMag> *accuracy_vec) const;

  virtual void getNodes(Teuchos::Array<double> *time_vec) const;
  virtual void removeNodes(Teuchos::Array<double> &time_vec);

  virtual int getOrder() const;

  virtual Rythmos::TimeRange<double> getTimeRange() const;

  //
  // Overriden from Rythmos::StepperBase<double>
  //

  virtual bool supportsCloning() const;
  virtual Teuchos::RCP<Rythmos::StepperBase<double> > cloneStepperAlgorithm() const;

  virtual bool isImplicit() const;

  virtual bool acceptsModel() const;
  virtual void setModel(const Teuchos::RCP<const Thyra::ModelEvaluator<double> > &model);
  virtual void setNonconstModel(const Teuchos::RCP<Thyra::ModelEvaluator<double> > &model);
  virtual bool modelIsConst() const;
  virtual Teuchos::RCP<const Thyra::ModelEvaluator<double> > getModel() const;
  virtual Teuchos::RCP<Thyra::ModelEvaluator<double> > getNonconstModel();

  virtual void setInitialCondition(const Thyra::ModelEvaluatorBase::InArgs<double>& initialCondition);
  virtual Thyra::ModelEvaluatorBase::InArgs<double> getInitialCondition() const;

  virtual double takeStep(double dt, Rythmos::StepSizeType stepType);
  virtual const Rythmos::StepStatus<double> getStepStatus() const;
  virtual void setStepControlData(const Rythmos::StepperBase<double> &stepper);

private:
  Teuchos::RCP<const Rythmos::StepperBase<double> > wrappedStepper_;
  Teuchos::RCP<const ReducedSpace> reducedSpace_;
  Teuchos::RCP<const Epetra_Map> fullMap_;

  void failNonconstFunction();
};

} // namespace Albany

#endif /*ALBANY_RYTHMOSSTEPPERFULLSTATEWRAPPER_HPP*/
