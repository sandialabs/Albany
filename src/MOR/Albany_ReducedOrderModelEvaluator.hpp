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

#ifndef ALBANY_REDUCEDORDERMODELEVALUATOR_HPP
#define ALBANY_REDUCEDORDERMODELEVALUATOR_HPP

#include "EpetraExt_ModelEvaluator.h"

#include "Teuchos_RCP.hpp"

namespace Albany {

class ReducedSpace;
class ReducedOperatorFactory;

class ReducedOrderModelEvaluator : public EpetraExt::ModelEvaluator {
public:
  ReducedOrderModelEvaluator(const Teuchos::RCP<EpetraExt::ModelEvaluator> &fullOrderModel,
                             const Teuchos::RCP<const ReducedSpace> &solutionSpace,
                             const Teuchos::RCP<ReducedOperatorFactory> &reducedOpFactory);

  // Overriden functions
  virtual Teuchos::RCP<const Epetra_Map> get_x_map() const;
  virtual Teuchos::RCP<const Epetra_Map> get_f_map() const;
  virtual Teuchos::RCP<const Epetra_Map> get_p_map(int l) const;
  virtual Teuchos::RCP<const Teuchos::Array<std::string> > get_p_names(int l) const;
  virtual Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

  virtual Teuchos::RCP<const Epetra_Vector> get_x_init() const;
  virtual Teuchos::RCP<const Epetra_Vector> get_x_dot_init() const;
  virtual Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;
  virtual double get_t_init() const;

  virtual double getInfBound() const;
  virtual Teuchos::RCP<const Epetra_Vector> get_p_lower_bounds(int l) const;
  virtual Teuchos::RCP<const Epetra_Vector> get_p_upper_bounds(int l) const;
  virtual double get_t_upper_bound() const;
  virtual double get_t_lower_bound() const;

  virtual Teuchos::RCP<Epetra_Operator> create_W() const;

  virtual InArgs createInArgs() const;
  virtual OutArgs createOutArgs() const;

  virtual void evalModel(const InArgs &inArgs, const OutArgs &outArgs) const;

  // Added functions
  void reset_x_and_x_dot_init();
  void reset_x_init();
  void reset_x_dot_init();

private:
  Teuchos::RCP<EpetraExt::ModelEvaluator> fullOrderModel_;
  Teuchos::RCP<const ReducedSpace> solutionSpace_;

  Teuchos::RCP<ReducedOperatorFactory> reducedOpFactory_;

  const Epetra_Map &componentMap() const;
  Teuchos::RCP<const Epetra_Map> componentMapRCP() const;

  Teuchos::RCP<Epetra_Vector> x_init_;
  Teuchos::RCP<Epetra_Vector> x_dot_init_;

  // Disallow copy and assignment
  ReducedOrderModelEvaluator(const ReducedOrderModelEvaluator &);
  ReducedOrderModelEvaluator &operator=(const ReducedOrderModelEvaluator &);
};

} // end namespace Albany

#endif /* ALBANY_REDUCEDORDERMODELEVALUATOR_HPP */
