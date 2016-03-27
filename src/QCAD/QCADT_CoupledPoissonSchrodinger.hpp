//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCADT_COUPLEDPOISSONSCHRODINGER_H
#define QCADT_COUPLEDPOISSONSCHRODINGER_H

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
namespace QCADT {
  std::string strdim(const std::string s, const int dim);
}


namespace QCADT {

/** \brief Thyra-based Model Evaluator for QCAD coupled poisson-schrodinger solver
 *
 */

  class CoupledPoissonSchrodinger : public Thyra::ModelEvaluatorDefaultBase<ST> {
  public:

    /** \name Constructors/initializers */
    //@{

      CoupledPoissonSchrodinger(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
				const Teuchos::RCP<const Teuchos_Comm>& comm, 
				const Teuchos::RCP<const Tpetra_Vector>& initial_guess, 
                                Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const> const &solver_factory);
    //@}

    ~CoupledPoissonSchrodinger();

    Teuchos::RCP<const Thyra::VectorSpaceBase<ST>> get_x_space() const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST>> get_f_space() const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST>> get_p_space(int l) const;
    Teuchos::RCP<const Thyra::VectorSpaceBase<ST>> get_g_space(int j) const;

    Teuchos::RCP<const Teuchos::Array<std::string> > get_p_names(int l) const;
    Teuchos::ArrayView<const std::string> get_g_names(int j) const
    { TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "not impl'ed"); }

    Thyra::ModelEvaluatorBase::InArgs<ST> getNominalValues() const;

    Thyra::ModelEvaluatorBase::InArgs<ST> getLowerBounds() const;
    Thyra::ModelEvaluatorBase::InArgs<ST> getUpperBounds() const;

    Teuchos::RCP<Thyra::LinearOpBase<ST> > create_W_op() const;
    Teuchos::RCP<Thyra::PreconditionerBase<ST> > create_W_prec() const;
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const> get_W_factory() const;
    
    Teuchos::RCP<Thyra::VectorSpaceBase<ST> const> createCombinedRangeSpace() const; 


    Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DgDx_op_impl(int j) const;
    Teuchos::RCP<Thyra::LinearOpBase<ST> > create_DgDx_dot_op_impl(int j) const;

    Thyra::ModelEvaluatorBase::InArgs<ST> createInArgs() const;
    Thyra::ModelEvaluatorBase::OutArgs<ST> createOutArgsImpl() const;

    void reportFinalPoint(
      Thyra::ModelEvaluatorBase::InArgs<ST> const & final_point,
      bool const was_solved);

    void allocateVectors(); 

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
    Teuchos::RCP<const Tpetra_Map> dist_eigenval_map;
    Teuchos::RCP<const Tpetra_Map> local_eigenval_map;
    Teuchos::RCP<Tpetra_Vector> eigenvals;
    Teuchos::RCP<Tpetra_MultiVector> x_schrodinger;
    Teuchos::RCP<Tpetra_Map> combined_SP_map;
    Teuchos::RCP<const Tpetra_Vector> saved_initial_guess;
    Thyra::ModelEvaluatorBase::InArgs<ST> nominal_values_; 

    Teuchos::RCP<Tpetra_CrsMatrix> Jac_Poisson; 
    Teuchos::RCP<Tpetra_CrsMatrix> Jac_Schrodinger; 

    //for setting get_W_factory() 
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const> solver_factory_;

    Teuchos::RCP<Albany::Application> poissonApp, schrodingerApp;
    Teuchos::RCP<Thyra::ModelEvaluator<ST> > poissonModel, schrodingerModel;

    Teuchos::RCP<const Teuchos_Comm> myComm;

    int nEigenvals;
    int my_nEigenvals_;  
    int num_param_vecs, num_response_vecs;
    int num_poisson_param_vecs, num_schrodinger_param_vecs;
    int num_models_; 

    double offset_to_CB; // conduction band = offset_to_CB - poisson_solution

    //! Material database
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

    //! Miscellaneous
    int numDims;
    double temperature;
    double length_unit_in_m;
    double energy_unit_in_eV;
    std::string quantumMtrlName; 
    int valleyDegeneracyFactor; 
    double effMass; 

    //! Sacado parameter vectors
    mutable Teuchos::Array<ParamVec> poisson_sacado_param_vec, schrodinger_sacado_param_vec;

    //! Element discretization (just for collected exodus output)
    Teuchos::RCP<Albany::AbstractDiscretization> disc;

    bool bVerbose;


  };
}
#endif
