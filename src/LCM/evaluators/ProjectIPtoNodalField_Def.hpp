//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include <Teuchos_TestForException.hpp>
#include "Albany_Utils.hpp"
#include "Adapt_NodalDataVector.hpp"

#include "Thyra_VectorBase.hpp"
#include <Thyra_TpetraMultiVector.hpp>
//#include <Thyra_LinearOpBase_decl.hpp>
//#include <Thyra_LinearOpWithSolveFactory_decl.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include "Thyra_LinearOpWithSolveBase.hpp"
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_MultiVectorStdOps.hpp>

//#ifdef HAVE_MPI
#  include "Tpetra_MpiPlatform.hpp"
//#else
//#  include "Tpetra_SerialPlatform.hpp"
//#endif

#ifdef ALBANY_MUELU
#include <Thyra_MueLuPreconditionerFactory.hpp>
#include "Stratimikos_MueluTpetraHelpers.hpp"
#endif

#ifdef ALBANY_IFPACK2
#include <Thyra_Ifpack2PreconditionerFactory.hpp>
#endif



namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ProjectIPtoNodalFieldBase<EvalT, Traits>::
ProjectIPtoNodalFieldBase(Teuchos::ParameterList& p,
		  const Teuchos::RCP<Albany::Layouts>& dl) :
    wBF   (p.get<std::string>                   ("Weighted BF Name"), dl->node_qp_scalar),
     BF   (p.get<std::string>                   ("BF Name"), dl->node_qp_scalar)

{

  this->addDependentField(wBF);
  this->addDependentField(BF);

  //! get and validate ProjectIPtoNodalField parameter list
  Teuchos::ParameterList* plist =
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidProjectIPtoNodalFieldParameters();
  plist->validateParameters(*reflist,0);

  output_to_exodus_ = plist->get<bool>("Output to File", true);

  //! number of quad points per cell and dimension
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;
  Teuchos::RCP<PHX::DataLayout> cell_dl = dl->cell_scalar;
  Teuchos::RCP<PHX::DataLayout> node_dl = dl->node_qp_vector;
  Teuchos::RCP<PHX::DataLayout> vert_vector_dl = dl->vertices_vector;
  num_pts_ = vector_dl->dimension(1);
  num_dims_ = vector_dl->dimension(2);
  num_nodes_ = node_dl->dimension(1);
  num_vertices_ = vert_vector_dl->dimension(2);

  //! Register with state manager
  this->p_state_mgr_ = p.get< Albany::StateManager* >("State Manager Ptr");

  // loop over the number of fields and register
  // Number of Fields is read off the input file - this is the number of named fields (scalar, vector, or tensor) to transfer
  number_of_fields_ = plist->get<int>("Number of Fields", 0);

  // resize field vectors
  ip_field_names_.resize(number_of_fields_);
  ip_field_layouts_.resize(number_of_fields_);
  nodal_field_names_.resize(number_of_fields_);
  ip_fields_.resize(number_of_fields_);

  for (int field(0); field < number_of_fields_; ++field) {
    ip_field_names_[field] = plist->get<std::string>(Albany::strint("IP Field Name", field));
    ip_field_layouts_[field] = plist->get<std::string>(Albany::strint("IP Field Layout", field));
    nodal_field_names_[field] = "proj_nodal_" + ip_field_names_[field];

    if (ip_field_layouts_[field] == "Scalar") {
      PHX::MDField<ScalarT> s(ip_field_names_[field], dl->qp_scalar);
      ip_fields_[field] = s;
    } else if (ip_field_layouts_[field] == "Vector") {
      PHX::MDField<ScalarT> v(ip_field_names_[field], dl->qp_vector);
      ip_fields_[field] = v;
    } else if (ip_field_layouts_[field] == "Tensor") {
      PHX::MDField<ScalarT> t(ip_field_names_[field], dl->qp_tensor);
      ip_fields_[field] = t;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Field Layout unknown");
    }

    // incoming integration point field to transfer
    this->addDependentField(ip_fields_[field]);

    if (ip_field_layouts_[field] == "Scalar" ) {
      this->p_state_mgr_->registerNodalVectorStateVariable(nodal_field_names_[field],
                                                dl->node_node_scalar,
                                                dl->dummy, "all",
                                                "scalar", 0.0, false,
                                                output_to_exodus_);
    } else if (ip_field_layouts_[field] == "Vector" ) {
      this->p_state_mgr_->registerNodalVectorStateVariable(nodal_field_names_[field],
                                                dl->node_node_vector,
                                                dl->dummy, "all",
                                                "scalar", 0.0, false,
                                                output_to_exodus_);
    } else if (ip_field_layouts_[field] == "Tensor" ) {
      this->p_state_mgr_->registerNodalVectorStateVariable(nodal_field_names_[field],
                                                dl->node_node_tensor,
                                                dl->dummy, "all",
                                                "scalar", 0.0, false,
                                                output_to_exodus_);
    }
  }

  // COunt the total number of vectors in the multivector
  num_vecs_ = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getVecsize();

  // Create field tag
  field_tag_ =
    Teuchos::rcp(new PHX::Tag<ScalarT>("Project IP to Nodal Field", dl->dummy));

  // Setup linear solver

#ifdef ALBANY_MUELU
    {
      Thyra::addMueLuToStratimikosBuilder(this->linearSolverBuilder); 
      Stratimikos::enableMueLuTpetra<LO, GO, KokkosNode>(linearSolverBuilder,"MueLu-Tpetra");
    }
#endif // MUELU

#ifdef ALBANY_IFPACK2
    {
      typedef Thyra::PreconditionerFactoryBase<ST> Base;
      typedef Thyra::Ifpack2PreconditionerFactory<Tpetra::CrsMatrix<ST, LO, GO, KokkosNode> > Impl;

      this->linearSolverBuilder.setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
    }
#endif // IFPACK2

  bool have_solver_pl = plist->isSublist("Solver Options");
  if(have_solver_pl){

    Teuchos::ParameterList &solver_list = plist->sublist("Solver Options");
    this->linearSolverBuilder.setParameterList(Teuchos::rcpFromRef(solver_list));

  }
  else {

    Teuchos::RCP<Teuchos::ParameterList> solver_list = Teuchos::rcp(new Teuchos::ParameterList);
    this->setDefaultSolverParameters(solver_list);
    this->linearSolverBuilder.setParameterList(solver_list);

  }

  this->lowsFactory = createLinearSolveStrategy(this->linearSolverBuilder);
  this->lowsFactory->setVerbLevel(Teuchos::VERB_LOW);
//  this->lowsFactory->setVerbLevel(Teuchos::VERB_EXTREME);


  this->addEvaluatedField(*field_tag_);

}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ProjectIPtoNodalFieldBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{

  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);

  for (int field(0); field < number_of_fields_; ++field) {
    this->utils.setFieldData(ip_fields_[field],fm);
  }
}

//------------------------------------------------------------------------------
// Specialization: Residual
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template<typename Traits>
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
ProjectIPtoNodalField(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ProjectIPtoNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
{
}

//------------------------------------------------------------------------------
template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  Teuchos::RCP<Adapt::NodalDataVector> node_data =
    this->p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getNodalDataVector();
  node_data->initializeVectors(0.0);

  Teuchos::RCP<Tpetra_CrsGraph> currentGraph =
    this->p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getNodalGraph();

//  Teuchos::RCP<const Tpetra_Map> nodeMap = node_data->getOverlapMap();
  Teuchos::RCP<const Tpetra_Map> nodeMap = node_data->getLocalMap();

//  if(Teuchos::is_null(this->mass_matrix) || !this->mass_matrix->getCrsGraph()->checkSizes(*currentGraph)){
  if(Teuchos::is_null(this->mass_matrix) || !currentGraph->checkSizes(*this->mass_matrix->getCrsGraph())){

     // reallocate the mass matrix

     this->mass_matrix = Teuchos::rcp(new Tpetra_CrsMatrix(currentGraph));
     this->source_load_vector = Teuchos::rcp(new Tpetra_MultiVector(nodeMap, this->num_vecs_, true));
     this->node_projected_ip_vector = Teuchos::rcp(new Tpetra_MultiVector(nodeMap, this->num_vecs_, false));

  }
  else {

     this->mass_matrix->resumeFill();

     // Zero the solution and mass matrix in preparation for summation / solution operations
     this->mass_matrix->setAllToScalar(0.0);
     this->source_load_vector->putScalar(0.0);

  }
}

//------------------------------------------------------------------------------
template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
fillMassMatrixFull (const typename Traits::EvalData& workset)
{
  const std::size_t
    num_nodes = this->num_nodes_,
    num_pts   = this->num_pts_;
  for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t rnode = 0; rnode < num_nodes; ++rnode) {
      GO global_row = workset.wsElNodeID[cell][rnode];
      Teuchos::Array<GO> cols;
      Teuchos::Array<ST> vals;

      for (std::size_t cnode = 0; cnode < num_nodes; ++cnode) {
        const GO global_col = workset.wsElNodeID[cell][cnode];
        cols.push_back(global_col);

        ST mass_value = 0;
        for (std::size_t qp = 0; qp < num_pts; ++qp)
          mass_value += this->wBF(cell, rnode, qp) * this->BF(cell, cnode, qp);
        vals.push_back(mass_value);
      }
      const LO ret =
        this->mass_matrix->sumIntoGlobalValues(global_row, cols, vals);
      TEUCHOS_TEST_FOR_EXCEPTION(
        ret != cols.size(), std::logic_error,
        "global_row " << global_row << " of mass_matrix is missing elements"
        << std::endl);
    }
  }
}

template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
fillMassMatrixLumped (const typename Traits::EvalData& workset)
{
  const std::size_t
    num_nodes = this->num_nodes_,
    num_pts   = this->num_pts_;
  for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp = 0; qp < num_pts; ++qp) {
      double diag = 0;
      for (std::size_t rnode = 0; rnode < num_nodes; ++rnode)
        diag += this->BF(cell, rnode, qp);
      for (std::size_t rnode = 0; rnode < num_nodes; ++rnode) {
        const GO global_row = workset.wsElNodeID[cell][rnode];
        const Teuchos::Array<GO> cols(1, workset.wsElNodeID[cell][rnode]);
        const Teuchos::Array<ST> vals(1, this->wBF(cell, rnode, qp) * diag);
        this->mass_matrix->replaceGlobalValues(global_row, cols, vals);
      }
    }
  }
}

template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
fillRHS (const typename Traits::EvalData& workset)
{
  Teuchos::RCP<Adapt::NodalDataVector> node_data =
    this->p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getNodalDataVector();
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >&  wsElNodeID = workset.wsElNodeID;

  const std::size_t
    num_nodes = this->num_nodes_,
    num_dims  = this->num_dims_,
    num_pts   = this->num_pts_;

  //todo Could optimize ip_field_layouts_[field] string check by moving to field
  // loop.
  for (std::size_t field = 0; field < this->number_of_fields_; ++field) {
    int node_var_offset;
    int node_var_ndofs;
    node_data->getNDofsAndOffset(this->nodal_field_names_[field], node_var_offset, node_var_ndofs);
    for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t node = 0; node < num_nodes; ++node) {
        GO global_row = wsElNodeID[cell][node];
        for (std::size_t qp = 0; qp < num_pts; ++qp) {
          if (this->ip_field_layouts_[field] == "Scalar" ) {
            // save the scalar component
            this->source_load_vector->sumIntoGlobalValue(
              global_row, node_var_offset,
              this->ip_fields_[field](cell, qp) * this->wBF(cell, node, qp));
          } else if (this->ip_field_layouts_[field] == "Vector" ) {
            for (std::size_t dim0 = 0; dim0 < num_dims; ++dim0) {
              // save the vector component
              this->source_load_vector->sumIntoGlobalValue(
                global_row, node_var_offset + dim0,
                this->ip_fields_[field](cell, qp, dim0) * this->wBF(cell, node, qp));
            }
          } else if (this->ip_field_layouts_[field] == "Tensor" ) {
            for (std::size_t dim0 = 0; dim0 < num_dims; ++dim0) {
              for (std::size_t dim1 = 0; dim1 < num_dims; ++dim1) {
                // save the tensor component
                this->source_load_vector->sumIntoGlobalValue(
                  global_row, node_var_offset + dim0*num_dims + dim1,
                  this->ip_fields_[field](cell, qp, dim0, dim1) * this->wBF(cell, node, qp));
              }
            }
          }
        }
      }
    } // end cell loop
  } // end field loop
}

template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the mass matrix.
  // Assumes: mass_matrix is the right size and ready to fill.
  //fillMassMatrixLumped(workset); //todo Include an option for this.
  fillMassMatrixFull(workset);

  // Deal with each of the fields in the multivector that stores the RHS of the
  // projection.
  fillRHS(workset);
}
//------------------------------------------------------------------------------

template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{

  typedef Teuchos::ScalarTraits<ST>::magnitudeType MT;  // Magnitude-type typedef
  ST zero = Teuchos::ScalarTraits<ST>::zero();
  ST one = Teuchos::ScalarTraits<ST>::one();

  Teuchos::RCP<Teuchos::FancyOStream>
    out = Teuchos::VerboseObjectBase::getDefaultOStream();

  // Note: we are in postEvaluate so all PEs call this

  // Get the node data vector container
  Teuchos::RCP<Adapt::NodalDataVector> node_data =
    this->p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getNodalDataVector();

  this->mass_matrix->fillComplete();

  // Do the solve
  // Create a Thyra linear operator (A) using the Tpetra::CrsMatrix (tpetra_A).

  const Teuchos::RCP<Tpetra::Operator<ST, LO, GO, KokkosNode> >
    tpetra_A = this->mass_matrix;

  const Teuchos::RCP<Thyra::LinearOpBase<ST> > A =
    Thyra::createLinearOp(tpetra_A);

  std::vector<MT> norm_b_vec(this->num_vecs_);
  std::vector<MT> norm_res_vec(this->num_vecs_);
  Teuchos::ArrayView<MT> norm_res = Teuchos::arrayViewFromVector(norm_res_vec);
  Teuchos::ArrayView<MT> norm_b = Teuchos::arrayViewFromVector(norm_b_vec);

  // Whether the linear solver succeeded.
  // (this will be set during the residual check at the end)
  bool success = true;
  MT              maxResid               = 1e-5;

  // Create a BelosLinearOpWithSolve object from the Belos LOWS factory.
  Teuchos::RCP<Thyra::LinearOpWithSolveBase<ST> >
    nsA = this->lowsFactory->createOp();

  // Initialize the BelosLinearOpWithSolve object with the Thyra linear operator.
  Thyra::initializeOp<ST>( *this->lowsFactory, A, nsA.ptr() );

  this->node_projected_ip_vector->putScalar(0.0);

  Teuchos::RCP< Thyra::MultiVectorBase<ST> >
    x = Thyra::createMultiVector(this->node_projected_ip_vector);

  Teuchos::RCP< Thyra::MultiVectorBase<ST> >
    b = Thyra::createMultiVector(this->source_load_vector);

  // Compute the column norms of the right-hand side b. If b = 0, no need to proceed.
  Thyra::norms_2( *b, norm_b );
  bool b_is_zero = true; 
  *out << "Norm of the b coming in" << std::endl;
  for (int i=0; i<this->num_vecs_; ++i) {
    *out << "b " << i+1 << " : "
	 << std::setw(16) << std::right << norm_b[i] << std::endl;
    if(norm_b[i] > 1.0e-16) b_is_zero = false;
  }
  if(b_is_zero) return;

  // Perform solve using the linear operator to get the approximate solution of Ax=b,
  // where b is the right-hand side and x is the left-hand side.

  Thyra::SolveStatus<ST> solveStatus = Thyra::solve( *nsA, Thyra::NOTRANS, *b, x.ptr() );

  // Print out status of solve.
    *out << "\nBelos LOWS Status: "<< solveStatus << std::endl;

  //
  // Compute residual and ST check convergence.
  Teuchos::RCP< Thyra::MultiVectorBase<ST> >
    y = Thyra::createMembers(x->range(), x->domain());

  Thyra::norms_2( *x, norm_b );
  *out << "Norm of the x going out" << std::endl;
  for (int i=0; i<this->num_vecs_; ++i) {
    *out << "RHS " << i+1 << " : "
	 << std::setw(16) << std::right << norm_b[i] << std::endl;
  }

  // Compute the column norms of the right-hand side b. If b = 0, no need to proceed.
  Thyra::norms_2( *b, norm_b );

  // Compute y=A*x, where x is the solution from the linear solver.
  A->apply(  Thyra::NOTRANS, *x, y.ptr(), 1.0, 0.0 );

  // Compute A*x-b = y-b
  Thyra::update( -one, *b, y.ptr() );
  Thyra::norms_2(*y, norm_res);

  // Print out the final relative residual norms.
  MT rel_res = 0.0;
  *out << "Final relative residual norms" << std::endl;
  for (int i=0; i<this->num_vecs_; ++i) {
    rel_res = norm_res[i]/norm_b[i];
    if (rel_res > maxResid)
      success = false;
    *out << "RHS " << i+1 << " : "
	 << std::setw(16) << std::right << rel_res << std::endl;
  }

  // Store the overlapped vector data back in stk in the field "field_name"
  node_data->saveNodalDataState(this->node_projected_ip_vector);

}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
ProjectIPtoNodalFieldBase<EvalT,Traits>::getValidProjectIPtoNodalFieldParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ProjectIPtoNodalField Params"));;

  // Dont validate the solver parameters used in the projection solve - let Stratimikos do it

  validPL->sublist("Solver Options").disableRecursiveValidation();

  validPL->set<std::string>("Name", "", "Name of field Evaluator");
  validPL->set<int>("Number of Fields", 0);
  validPL->set<std::string>("IP Field Name 0", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 0", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 1", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 1", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 2", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 2", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 3", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 3", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 4", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 4", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 5", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 5", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 6", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 6", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 7", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 7", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 8", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 8", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<std::string>("IP Field Name 9", "", "IP Field prefix");
  validPL->set<std::string>("IP Field Layout 9", "", "IP Field Layout: Scalar, Vector, or Tensor");

  validPL->set<bool>("Output to File", true, "Whether nodal field info should be output to a file");
  validPL->set<bool>("Generate Nodal Values", true, "Whether values at the nodes should be generated");

  return validPL;
}


template<typename EvalT, typename Traits>
void
ProjectIPtoNodalFieldBase<EvalT,Traits>::setDefaultSolverParameters(const Teuchos::RCP<Teuchos::ParameterList>& pl) const
{

  pl->set<std::string>("Linear Solver Type", "Belos");

  Teuchos::ParameterList& solver_types =
    pl->sublist("Linear Solver Types");

  Teuchos::ParameterList& belos_types =
    solver_types.sublist("Belos");

  //
  // Set the parameters for the Belos LOWS Factory and create a parameter list.
  //
  int             blockSize              = 1;
  int             maxIterations          = 1000;
  int             maxRestarts            = 15;
  int             gmresKrylovLength      = 50;
  int             outputFrequency        = 100;
  bool            outputMaxResOnly       = true;
  double          maxResid               = 1e-5;

  // The mass matrix is symmetric positive definite, so we can use CG rather
  // than GMRES.
  belos_types.set<std::string>("Solver Type", "Block CG");

  Teuchos::ParameterList& belosLOWSFPL_solver =
    belos_types.sublist("Solver Types");

  Teuchos::ParameterList& belosLOWSFPL_gmres =
    belosLOWSFPL_solver.sublist("Block GMRES");

  belosLOWSFPL_gmres.set<int>("Maximum Iterations",maxIterations);
  belosLOWSFPL_gmres.set<double>("Convergence Tolerance",maxResid);
  belosLOWSFPL_gmres.set<int>("Maximum Restarts",maxRestarts);
  belosLOWSFPL_gmres.set<int>("Block Size",blockSize);
  belosLOWSFPL_gmres.set<int>("Num Blocks",gmresKrylovLength);
  belosLOWSFPL_gmres.set<int>("Output Frequency",outputFrequency);
  belosLOWSFPL_gmres.set<bool>("Show Maximum Residual Norm Only",outputMaxResOnly);

//  pl->set<std::string>("Preconditioner Type", "Ifpack2");
  pl->set<std::string>("Preconditioner Type", "None");

/*
  Teuchos::ParameterList& prec_types =
    pl->sublist("Preconditioner Types");

  Teuchos::ParameterList& ifpack_types =
    prec_types.sublist("Ifpack2");

  ifpack_types.set<int>("Overlap",2);
  ifpack_types.set<std::string>("Prec Type", "ILUT");

  Teuchos::ParameterList& ifpack_settings =
    ifpack_types.sublist("Ifpack2 Settings");
  ifpack_settings.set<double>("fact: drop tolerance", 0);
  ifpack_settings.set<double>("fact: ilut level-of-fill", 1);
  ifpack_settings.set<int>("fact: level-of-fill", 1);
*/

}

//------------------------------------------------------------------------------
}

