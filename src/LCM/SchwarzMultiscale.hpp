//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzMultiscale_hpp)
#define LCM_SchwarzMultiscale_hpp

#include "Albany_ModelEvaluatorT.hpp"
#include "Albany_DataTypes.hpp"
#include "Schwarz_BoundaryJacobian.hpp" 

namespace LCM {

///
/// \brief Definition for the SchwarzMultiscale
///
class SchwarzMultiscale: public Thyra::ModelEvaluatorDefaultBase<ST> {

public:

  /// Constructor
  SchwarzMultiscale(Teuchos::RCP<Teuchos::ParameterList> const & app_params,
      Teuchos::RCP<Teuchos::Comm<int> const > const & commT,
      Teuchos::RCP<Tpetra_Vector const > const & initial_guessT, 
      const Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST> > & solver_factory);

  ///Destructor
  ~SchwarzMultiscale();

  /// Return solution vector map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_x_space() const;

  /// Return residual vector map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_f_space() const;

  /// Return parameter vector map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_p_space(int l) const;

  /// Return response function map
  Teuchos::RCP<Thyra::VectorSpaceBase<ST> const>
  get_g_space(int j) const;

  /// Return array of parameter names
  Teuchos::RCP<Teuchos::Array<std::string> const>
  get_p_names(int l) const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getNominalValues() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getLowerBounds() const;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  getUpperBounds() const;

  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  create_W_op() const;

  /// Create preconditioner operator
  Teuchos::RCP<Thyra::PreconditionerBase<ST> >
  create_W_prec() const;

  Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<ST> const>
  get_W_factory() const;

  /// Create InArgs
  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgs() const;

  void
  reportFinalPoint(
      Thyra::ModelEvaluatorBase::InArgs<ST> const & final_point,
      bool const was_solved);

  void
  allocateVectors();

  /// Create the map for the coupled solution from an array of the maps
  /// for each individual model that is being coupled.
  Teuchos::RCP<Tpetra_Map const>
  createCoupledMap(
      Teuchos::Array<Teuchos::RCP<Tpetra_Map const> > maps,
      Teuchos::RCP<Teuchos::Comm<int> const> const & commT);

  //Take in a combined vector for a coupled model 
  //and separates into into individual subvectors for each submodel.
  //These are stored in a Teuchos::Array of Tpetra_Vectors.
  void separateCoupledVectorConst( const Teuchos::RCP<const Tpetra_Vector>& combined_vector, 
                              Teuchos::Array<Teuchos::RCP<const Tpetra_Vector> >& vecs) const;
  
  void separateCoupledVectorNonConst( const Teuchos::RCP<Tpetra_Vector>& combined_vector, 
                              Teuchos::Array<Teuchos::RCP<Tpetra_Vector> >& vecs) const;

protected:

  /// Create operator form of dg/dx for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  create_DgDx_op_impl(int j) const;

  /// Create operator form of dg/dx_dot for distributed responses
  Teuchos::RCP<Thyra::LinearOpBase<ST> >
  create_DgDx_dot_op_impl(int j) const;

  /// Create OutArgs
  Thyra::ModelEvaluatorBase::OutArgs<ST>
  createOutArgsImpl() const;

  /// Evaluate model on InArgs
  void
  evalModelImpl(
      Thyra::ModelEvaluatorBase::InArgs<ST> const & in_args,
      Thyra::ModelEvaluatorBase::OutArgs<ST> const & out_args) const;

private:
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidAppParameters() const;

  Teuchos::RCP<Teuchos::ParameterList const>
  getValidProblemParameters() const;

  /// RCP to matDB object
  Teuchos::Array<Teuchos::RCP<QCAD::MaterialDatabase> >
  material_dbs_;

  Teuchos::Array<Teuchos::RCP<Thyra::ModelEvaluator<ST> > >
  models_;

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> >
  apps_;

  Teuchos::RCP<Teuchos::Comm<int> const>
  commT_;

  /// Cached nominal values -- this contains stuff like x_init, x_dot_init, etc.
  Thyra::ModelEvaluatorBase::InArgs<ST>
  nominal_values_;

  Thyra::ModelEvaluatorBase::InArgs<ST>
  createInArgsImpl() const;

  Teuchos::RCP<const Tpetra_Map>
  coupled_disc_map_;
  
  Teuchos::Array<Teuchos::RCP<Tpetra_Map const> >
  disc_maps_;

  //Teuchos array holding main diagonal jacobians (non-coupled models)
  Teuchos::Array<Teuchos::RCP<Tpetra_CrsMatrix> > jacs_;

  //Teuchos array holding off-diagonal jacobians (coupling ones)
  Teuchos::Array<Teuchos::RCP<LCM::Schwarz_BoundaryJacobian> > jacs_boundary_;

  int
  num_models_;

  Teuchos::Array<int>
  num_params_;

  Teuchos::Array<int>
  num_responses_;
  
  Teuchos::Array<int>
  num_params_partial_sum_;

  Teuchos::Array<int>
  num_responses_partial_sum_;
  
  //like num_param_vecs
  int
  num_params_total_;

  //like dist_param_vecs
  int
  num_dist_params_total_;

  //like num_response_vecs
  int
  num_responses_total_;

  //for setting get_W_factory() 
  Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<ST> > solver_factory_;
    
  mutable Teuchos::Array<Teuchos::Array<ParamVec> >
  sacado_param_vecs_;

  mutable Teuchos::Array<Thyra::ModelEvaluatorBase::InArgs<ST> >
  solver_inargs_;

  mutable Teuchos::Array<Thyra::ModelEvaluatorBase::OutArgs<ST> >
  solver_outargs_;

};

} // namespace LCM

#endif // LCM_SchwarzMultiscale_hpp
