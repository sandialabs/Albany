//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_COUPLEDPOISSONSCHRODINGERT_H
#define QCAD_COUPLEDPOISSONSCHRODINGERT_H

#include <iostream>

#include "LOCA.H"

#include "Albany_ModelEvaluatorT.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_Utils.hpp"
#include "Albany_DataTypes.hpp"

#include "QCAD_MaterialDatabase.hpp"
#include "Petra_Converters.hpp"

#include "Thyra_DefaultProductVector.hpp"
#include "Thyra_DefaultProductVectorSpace.hpp"


// Utility functions
namespace QCAD {
  std::string strdim(const std::string s, const int dim);
}


namespace QCAD {

/** \brief Thyra-based Model Evaluator for QCAD coupled poisson-schrodinger solver
 *
 */

  class CoupledPoissonSchrodingerT : public Thyra::ModelEvaluatorDefaultBase<ST> {
  public:

    /** \name Constructors/initializers */
    //@{

      CoupledPoissonSchrodingerT(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
				const Teuchos::RCP<const Teuchos_Comm>& comm, 
				const Teuchos::RCP<const Tpetra_Vector>& initial_guess);
    //@}

    ~CoupledPoissonSchrodingerT();

    Teuchos::RCP<const Thyra::VectorSpaceBase<ST>> get_x_space() const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST>> get_f_space() const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST>> get_p_space(int l) const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST>> get_g_space(int j) const;

    Teuchos::RCP<const Teuchos::Array<std::string> > get_p_names(int l) const;

    Thyra::ModelEvaluatorBase::InArgs<ST> getNominalValues() const;

    Thyra::ModelEvaluatorBase::InArgs<ST> getLowerBounds() const;
    Thyra::ModelEvaluatorBase::InArgs<ST> getUpperBounds() const;

    Teuchos::RCP<Thyra::LinearOpBase<ST> > create_W_op() const;
    Teuchos::RCP<Thyra::PreconditionerBase<ST> > create_W_prec() const;
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const> get_W_factory() const;


    Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DgDx_op_impl(int j) const;
    Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DgDx_dot_op_impl(int j) const;

    Thyra::ModelEvaluatorBase::InArgs<ST> createInArgs() const;
    Thyra::ModelEvaluatorBase::OutArgs<ST> createOutArgsImpl() const;

    void reportFinalPoint(
      Thyra::ModelEvaluatorBase::InArgs<ST> const & final_point,
      bool const was_solved);

    void evalModelImpl(
      Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
      Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const;

    Teuchos::RCP<Albany::Application> getPoissonApp() const;
    Teuchos::RCP<Albany::Application> getSchrodingerApp() const;

    Teuchos::RCP<Albany::AbstractDiscretization> getDiscretization() const { return disc; }

  /*public:  //as public for QCAD::Solver and QCAD::CoupledPSObserver to use -- maybe make these friends and declare private?
    void separateCombinedVector(const Teuchos::RCP<Epetra_Vector>& combinedVector,
				Teuchos::RCP<Epetra_Vector>& poisson_part,
				Teuchos::RCP<Epetra_MultiVector>& schrodinger_part) const;

    void separateCombinedVector(const Teuchos::RCP<Epetra_Vector>& combinedVector,
				Teuchos::RCP<Epetra_Vector>& poisson_part,
				Teuchos::RCP<Epetra_MultiVector>& schrodinger_part,
				Teuchos::RCP<Epetra_Vector>& eigenvalue_part) const;

    void separateCombinedVector(const Teuchos::RCP<const Epetra_Vector>& combinedVector,
				Teuchos::RCP<const Epetra_Vector>& poisson_part,
				Teuchos::RCP<const Epetra_MultiVector>& schrodinger_part) const;

    void separateCombinedVector(const Teuchos::RCP<const Epetra_Vector>& combinedVector,
				Teuchos::RCP<const Epetra_Vector>& poisson_part,
				Teuchos::RCP<const Epetra_MultiVector>& schrodinger_part,
				Teuchos::RCP<const Epetra_Vector>& eigenvalue_part) const;
    */

  private:
    Thyra::ModelEvaluatorBase::InArgs<ST>createInArgsImpl() const;
    Teuchos::RCP<const Teuchos::ParameterList> getValidAppParameters() const;
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;


    //Solely for testing Jacobian
    //void computeResidual(const Teuchos::RCP<const Epetra_Vector>& x,
    //			 Teuchos::RCP<Epetra_Vector>& f,
    //			 Teuchos::RCP<Epetra_CrsMatrix>& massMx) const;


  private:
    Teuchos::RCP<const Tpetra_Map> disc_map, disc_overlap_map;
    Teuchos::RCP<Tpetra_Map> combined_SP_map;
    Teuchos::RCP<const Tpetra_Vector> saved_initial_guess;
    Thyra::ModelEvaluatorBase::InArgs<ST> nominal_values_;


    Teuchos::RCP<Albany::Application> poissonApp, schrodingerApp;
    Teuchos::RCP<Teuchos::RCP<Thyra::ModelEvaluator<ST> > > poissonModel, schrodingerModel;

    Teuchos::RCP<const Teuchos_Comm> myComm;

    int nEigenvals;
    int num_param_vecs, num_response_vecs;
    int num_poisson_param_vecs, num_schrodinger_param_vecs;

    double offset_to_CB; // conduction band = offset_to_CB - poisson_solution

    //! Material database
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

    //! Miscellaneous
    int numDims;
    double temperature;
    double length_unit_in_m;
    double energy_unit_in_eV;

    //! Sacado parameter vectors
    mutable Teuchos::Array<ParamVec> poisson_sacado_param_vec, schrodinger_sacado_param_vec;

    //! Element discretization (just for collected exodus output)
    Teuchos::RCP<Albany::AbstractDiscretization> disc;

    bool bVerbose;


  };
}
#endif
