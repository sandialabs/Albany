//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_ADAPTIVEMODELEVALUATOR_HPP
#define AADAPT_ADAPTIVEMODELEVALUATOR_HPP

#include "EpetraExt_ModelEvaluator.h"
#include "Teuchos_RCP.hpp"

namespace AAdapt {

class AdaptiveModelEvaluator : public EpetraExt::ModelEvaluator {

  public:

//    AdaptiveModelEvaluator(const Teuchos::RCP<EpetraExt::ModelEvaluator>& oldCoarseModel,
//                           const Teuchos::RCP<const ReducedSpace>& solutionSpace,
//                           const Teuchos::RCP<ReducedOperatorFactory>& reducedOpFactory);
    AdaptiveModelEvaluator(const Teuchos::RCP<EpetraExt::ModelEvaluator>& oldCoarseModel);

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
    virtual Teuchos::RCP<Epetra_Operator> create_DgDp_op(int j, int l) const;

    virtual InArgs createInArgs() const;
    virtual OutArgs createOutArgs() const;

    virtual void evalModel(const InArgs& inArgs, const OutArgs& outArgs) const;

    // Added functions
    void reset_x_and_x_dot_init();
    void reset_x_init();
    void reset_x_dot_init();

  private:

    Teuchos::RCP<EpetraExt::ModelEvaluator> oldCoarseModel_;
//    Teuchos::RCP<const ReducedSpace> solutionSpace_;

//    Teuchos::RCP<ReducedOperatorFactory> reducedOpFactory_;

    const Epetra_Map& componentMap() const;
    Teuchos::RCP<const Epetra_Map> componentMapRCP() const;

    Teuchos::RCP<Epetra_Vector> x_init_;
    Teuchos::RCP<Epetra_Vector> x_dot_init_;

    // Disallow copy and assignment
    AdaptiveModelEvaluator(const AdaptiveModelEvaluator&);
    AdaptiveModelEvaluator& operator=(const AdaptiveModelEvaluator&);
};

} // namespace AAdapt

#endif /* ALBANY_ADAPTIVEMODELEVALUATOR_HPP */
