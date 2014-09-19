//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_AbstractFactoryStd.hpp>
#include "Albany_Utils.hpp"

#include "Thyra_VectorBase.hpp"
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include "Thyra_LinearOpWithSolveBase.hpp"

#ifdef ALBANY_IFPACK2
#include <Thyra_Ifpack2PreconditionerFactory.hpp>
#endif

namespace LCM {

template<typename EvalT, typename Traits>
typename ProjectIPtoNodalFieldBase<EvalT, Traits>::EFieldLayout::Enum
ProjectIPtoNodalFieldBase<EvalT, Traits>::EFieldLayout::
fromString (const std::string& str)
  throw (Teuchos::Exceptions::InvalidParameterValue)
{
  if (str == "Scalar") return scalar;
  else if (str == "Vector") return vector;
  else if (str == "Tensor") return tensor;
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameterValue,
      "Field Layout value " << str
      << "is invalid; valid values are Scalar, Vector, and Tensor.");
  }
}

namespace {
Teuchos::RCP<Teuchos::ParameterList> getValidProjectIPtoNodalFieldParameters ()
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    rcp(new Teuchos::ParameterList("Valid ProjectIPtoNodalField Params"));;

  // Dont validate the solver parameters used in the projection solve - let Stratimikos do it

  valid_pl->sublist("Solver Options").disableRecursiveValidation();

  valid_pl->set<std::string>("Name", "", "Name of field Evaluator");
  valid_pl->set<int>("Number of Fields", 0);

  for (int i = 0; i < 9; ++i) {
    valid_pl->set<std::string>(Albany::strint("IP Field Name", i), "",
                              "IP Field prefix");
    valid_pl->set<std::string>(Albany::strint("IP Field Layout", i), "",
                              "IP Field Layout: Scalar, Vector, or Tensor");
  }

  valid_pl->set<bool>("Output to File", true, "Whether nodal field info should be output to a file");
  valid_pl->set<bool>("Generate Nodal Values", true, "Whether values at the nodes should be generated");

  valid_pl->set<std::string>("Mass Matrix Type", "Full", "Full or Lumped");

  return valid_pl;
}

void setDefaultSolverParameters (Teuchos::ParameterList& pl)
{
  pl.set<std::string>("Linear Solver Type", "Belos");

  Teuchos::ParameterList& solver_types = pl.sublist("Linear Solver Types");
  Teuchos::ParameterList& belos_types = solver_types.sublist("Belos");
  belos_types.set<std::string>("Solver Type", "Block CG");

  Teuchos::ParameterList& solver = belos_types.sublist("Solver Types").sublist("Block CG");
  solver.set<int>("Maximum Iterations", 1000);
  //todo It's easy on the model problem to get this accuracy in ~10 CG
  // iterations, but how much accuracy do we want in practice?
  solver.set<double>("Convergence Tolerance", 1e-12);

#ifdef ALBANY_IFPACK2
  pl.set<std::string>("Preconditioner Type", "Ifpack2");
  Teuchos::ParameterList& prec_types = pl.sublist("Preconditioner Types");
  Teuchos::ParameterList& ifpack_types = prec_types.sublist("Ifpack2");

  ifpack_types.set<int>("Overlap", 0);
  // Both of these preconditioners are quite effective on the model
  // problem. I'll have to wait to see other problems before I decided whether:
  // (a) Diagonal is always sufficient; (b) ILU(0) with ovl=0 is good; or (c)
  // something more powerful is required (ILUTP or overlap > 0, say).
  const char* prec_type[] = {"RILUK", "Diagonal"};
  ifpack_types.set<std::string>("Prec Type", prec_type[0]);

  Teuchos::ParameterList& ifpack_settings = ifpack_types.sublist("Ifpack2 Settings");
  ifpack_settings.set<int>("fact: iluk level-of-fill", 0);
#endif
}

// Represent the mass matrix type by an enumerated type.
struct EMassMatrixType {
  enum Enum { full, lumped };
  static Enum fromString (const std::string& str)
    throw (Teuchos::Exceptions::InvalidParameterValue)
  {
    if (str == "Full") return full;
    else if (str == "Lumped") return lumped;
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(
        true, Teuchos::Exceptions::InvalidParameterValue,
        "Mass Matrix Type value " << str
        << "is invalid; valid values are Full and Lumped.");
    }
  }
};
} // namespace

template<typename Traits>
class ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::MassMatrix {
public:
  MassMatrix (
    const ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>* base)
    : base_(base) {}

  virtual void fill(const typename Traits::EvalData& workset) = 0;

  Teuchos::RCP<Tpetra_CrsMatrix>& matrix () { return matrix_; }

  static
  ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::MassMatrix*
  create(EMassMatrixType::Enum type,
         const ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>* base);
protected:
  const ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>* base_;
  Teuchos::RCP<Tpetra_CrsMatrix> matrix_;
};

template<typename Traits>
class ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::FullMassMatrix
  : public ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::MassMatrix {
public:
  FullMassMatrix (const ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>* base)
    : MassMatrix(base) {}
  virtual void fill (const typename Traits::EvalData& workset) {
    const std::size_t
      num_nodes = this->base_->num_nodes_,
      num_pts   = this->base_->num_pts_;
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
            mass_value +=
              this->base_->wBF(cell, rnode, qp) * this->base_->BF(cell, cnode, qp);
          vals.push_back(mass_value);
        }
        const LO ret = this->matrix_->sumIntoGlobalValues(global_row, cols, vals);
        TEUCHOS_TEST_FOR_EXCEPTION(
          ret != cols.size(), std::logic_error,
          "global_row " << global_row << " of mass matrix is missing elements"
          << std::endl);
      }
    }
  }
};

template<typename Traits>
class ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::LumpedMassMatrix
  : public ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::MassMatrix {
public:
  LumpedMassMatrix (const ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>* base)
    : MassMatrix(base) {}
  virtual void fill (const typename Traits::EvalData& workset) {
    const std::size_t
      num_nodes = this->base_->num_nodes_,
      num_pts   = this->base_->num_pts_;
    for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t rnode = 0; rnode < num_nodes; ++rnode) {
        const GO global_row = workset.wsElNodeID[cell][rnode];
        const Teuchos::Array<GO> cols(1, global_row);
        double diag = 0;
        for (std::size_t qp = 0; qp < num_pts; ++qp) {
          double diag_qp = 0;
          for (std::size_t cnode = 0; cnode < num_nodes; ++cnode)
            diag_qp += this->base_->BF(cell, cnode, qp);
          diag += this->base_->wBF(cell, rnode, qp) * diag_qp;
        }
        const Teuchos::Array<ST> vals(1, diag);
        this->matrix_->sumIntoGlobalValues(global_row, cols, vals);
      }
    }
  }
};

template<typename Traits>
typename ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::MassMatrix*
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::MassMatrix::
create (EMassMatrixType::Enum type,
        const ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>* base) {
  switch (type) {
  case EMassMatrixType::full:
    return new FullMassMatrix(base);
    break;
  case EMassMatrixType::lumped:
    return new LumpedMassMatrix(base);
    break;
  }
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ProjectIPtoNodalFieldBase<EvalT, Traits>::
ProjectIPtoNodalFieldBase (Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
    wBF(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
     BF(p.get<std::string>("BF Name"), dl->node_qp_scalar)
{
  this->addDependentField(wBF);
  this->addDependentField(BF);

  //! get and validate ProjectIPtoNodalField parameter list
  Teuchos::ParameterList* plist =
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    getValidProjectIPtoNodalFieldParameters();
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

  for (int field = 0; field < number_of_fields_; ++field) {
    ip_field_names_[field] = plist->get<std::string>(Albany::strint("IP Field Name", field));
    nodal_field_names_[field] = "proj_nodal_" + ip_field_names_[field];
    ip_field_layouts_[field] = EFieldLayout::fromString(
      plist->get<std::string>(Albany::strint("IP Field Layout", field)));

    Teuchos::RCP<PHX::DataLayout> qp_layout, node_node_layout;
    switch (ip_field_layouts_[field]) {
    case EFieldLayout::scalar:
      qp_layout = dl->qp_scalar;
      node_node_layout = dl->node_node_scalar;
      break;
    case EFieldLayout::vector:
      qp_layout = dl->qp_vector;
      node_node_layout = dl->node_node_vector;
      break;
    case EFieldLayout::tensor:
      qp_layout = dl->qp_tensor;
      node_node_layout = dl->node_node_tensor;
      break;
    }
    ip_fields_[field] = PHX::MDField<ScalarT>(ip_field_names_[field], qp_layout);
    // Incoming integration point field to transfer.
    this->addDependentField(ip_fields_[field]);
    this->p_state_mgr_->registerNodalVectorStateVariable(
      nodal_field_names_[field], node_node_layout, dl->dummy, "all", "scalar",
      0.0, false, output_to_exodus_);
  }

  // Count the total number of vectors in the multivector
  num_vecs_ = this->p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getVecsize();

  // Create field tag
  field_tag_ =
    Teuchos::rcp(new PHX::Tag<ScalarT>("Project IP to Nodal Field", dl->dummy));

  // Set up linear solver
#ifdef ALBANY_IFPACK2
  {
    typedef Thyra::PreconditionerFactoryBase<ST> Base;
    typedef Thyra::Ifpack2PreconditionerFactory<Tpetra::CrsMatrix<ST, LO, GO, KokkosNode> > Impl;

    this->linearSolverBuilder_.setPreconditioningStrategyFactory(Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
  }
#endif // IFPACK2

  {
    Teuchos::RCP<Teuchos::ParameterList> solver_list =
      Teuchos::rcp(new Teuchos::ParameterList);
    // Use what has been provided.
    if (plist->isSublist("Solver Options"))
      solver_list->setParameters(plist->sublist("Solver Options"));
    { // Set the rest of the parameters to their default values.
      Teuchos::ParameterList pl;
      setDefaultSolverParameters(pl);
      solver_list->setParametersNotAlreadySet(pl);
    }
    this->linearSolverBuilder_.setParameterList(solver_list);
  }

  this->lowsFactory_ = createLinearSolveStrategy(this->linearSolverBuilder_);
  this->lowsFactory_->setVerbLevel(Teuchos::VERB_LOW);

  this->addEvaluatedField(*field_tag_);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ProjectIPtoNodalFieldBase<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
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
ProjectIPtoNodalField (Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  ProjectIPtoNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
{
  Teuchos::ParameterList* pl =
    p.get<Teuchos::ParameterList*>("Parameter List");
  if (!pl->getPtr<std::string>("Mass Matrix Type"))
    pl->set<std::string>("Mass Matrix Type", "Full", "Full or Lumped");

  {
    EMassMatrixType::Enum mass_matrix_type;
    const std::string& mmstr = pl->get<std::string>("Mass Matrix Type");
    try {
      mass_matrix_type =
        EMassMatrixType::fromString(mmstr);
    } catch (const Teuchos::Exceptions::InvalidParameterValue& e) {
      *Teuchos::VerboseObjectBase::getDefaultOStream()
        << "Warning: Mass Matrix Type was set to " << mmstr
        << ", which is invalid; setting to Full." << std::endl;
      mass_matrix_type = EMassMatrixType::full;
    }
    mass_matrix_ = Teuchos::rcp(MassMatrix::create(mass_matrix_type, this));
  }
}

//------------------------------------------------------------------------------
template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate (typename Traits::PreEvalData workset)
{
  Teuchos::RCP<Adapt::NodalDataVector> node_data =
    this->p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getNodalDataVector();
  node_data->initializeVectors(0.0);

  Teuchos::RCP<Tpetra_CrsGraph> currentGraph =
    this->p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getNodalGraph();

  Teuchos::RCP<const Tpetra_Map> nodeMap = node_data->getLocalMap(); // node_data->getOverlapMap();

  if (Teuchos::is_null(this->mass_matrix_->matrix()) ||
      !currentGraph->checkSizes(*this->mass_matrix_->matrix()->getCrsGraph())){
    // reallocate the mass matrix
    this->mass_matrix_->matrix() = Teuchos::rcp(new Tpetra_CrsMatrix(currentGraph));
    this->source_load_vector_ = Teuchos::rcp(new Tpetra_MultiVector(nodeMap, this->num_vecs_, true));
    this->node_projected_ip_vector_ = Teuchos::rcp(new Tpetra_MultiVector(nodeMap, this->num_vecs_, false));
  }
  else {
    this->mass_matrix_->matrix()->resumeFill();
    // Zero the solution and mass matrix in preparation for summation / solution operations
    this->mass_matrix_->matrix()->setAllToScalar(0.0);
    this->source_load_vector_->putScalar(0.0);
  }
}

//------------------------------------------------------------------------------
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

  for (std::size_t field = 0; field < this->number_of_fields_; ++field) {
    int node_var_offset;
    int node_var_ndofs;
    node_data->getNDofsAndOffset(this->nodal_field_names_[field], node_var_offset, node_var_ndofs);
    for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t node = 0; node < num_nodes; ++node) {
        GO global_row = wsElNodeID[cell][node];
        for (std::size_t qp = 0; qp < num_pts; ++qp) {
          switch (this->ip_field_layouts_[field]) {
          case ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::EFieldLayout::scalar:
            // save the scalar component
            this->source_load_vector_->sumIntoGlobalValue(
              global_row, node_var_offset,
              this->ip_fields_[field](cell, qp) * this->wBF(cell, node, qp));
            break;
          case ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::EFieldLayout::vector:
            for (std::size_t dim0 = 0; dim0 < num_dims; ++dim0) {
              // save the vector component
              this->source_load_vector_->sumIntoGlobalValue(
                global_row, node_var_offset + dim0,
                this->ip_fields_[field](cell, qp, dim0) * this->wBF(cell, node, qp));
            }
            break;
          case ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::EFieldLayout::tensor:
            for (std::size_t dim0 = 0; dim0 < num_dims; ++dim0) {
              for (std::size_t dim1 = 0; dim1 < num_dims; ++dim1) {
                // save the tensor component
                this->source_load_vector_->sumIntoGlobalValue(
                  global_row, node_var_offset + dim0*num_dims + dim1,
                  this->ip_fields_[field](cell, qp, dim0, dim1) * this->wBF(cell, node, qp));
              }
            }
            break;
          }
        }
      }
    } // end cell loop
  } // end field loop
}

template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  // Volume averaged field. Store as nodal data that will be scattered and
  // summed.

  // Assume: mass_matrix_ is the right size and ready to fill.
  this->mass_matrix_->fill(workset);

  // Deal with each of the fields in the multivector that stores the RHS of the
  // projection.
  fillRHS(workset);
}

//------------------------------------------------------------------------------
template<typename Traits>
void ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
postEvaluate (typename Traits::PostEvalData workset)
{
  typedef Teuchos::ScalarTraits<ST>::magnitudeType MT;  // Magnitude-type typedef
  const ST zero = Teuchos::ScalarTraits<ST>::zero();
  const ST one = Teuchos::ScalarTraits<ST>::one();

  Teuchos::RCP<Teuchos::FancyOStream>
    out = Teuchos::VerboseObjectBase::getDefaultOStream();

  // Note: we are in postEvaluate so all PEs call this

  // Get the node data vector container
  Teuchos::RCP<Adapt::NodalDataVector> node_data =
    this->p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getNodalDataVector();

  this->mass_matrix_->matrix()->fillComplete();

  // Do the solve
  // Create a Thyra linear operator (A) using the Tpetra::CrsMatrix (tpetra_A).

  const Teuchos::RCP<Tpetra::Operator<ST, LO, GO, KokkosNode> >
    tpetra_A = this->mass_matrix_->matrix();

  const Teuchos::RCP<Thyra::LinearOpBase<ST> > A =
    Thyra::createLinearOp(tpetra_A);

  std::vector<MT> norm_b_vec(this->num_vecs_);
  std::vector<MT> norm_res_vec(this->num_vecs_);
  Teuchos::ArrayView<MT> norm_res = Teuchos::arrayViewFromVector(norm_res_vec);
  Teuchos::ArrayView<MT> norm_b = Teuchos::arrayViewFromVector(norm_b_vec);

  // Whether the linear solver succeeded.
  // (this will be set during the residual check at the end)
  bool success = true;
  const MT maxResid = 1e-5;

  // Create a BelosLinearOpWithSolve object from the Belos LOWS factory.
  Teuchos::RCP<Thyra::LinearOpWithSolveBase<ST> >
    nsA = this->lowsFactory_->createOp();

  // Initialize the BelosLinearOpWithSolve object with the Thyra linear operator.
  Thyra::initializeOp<ST>( *this->lowsFactory_, A, nsA.ptr() );

  this->node_projected_ip_vector_->putScalar(0.0);

  Teuchos::RCP< Thyra::MultiVectorBase<ST> >
    x = Thyra::createMultiVector(this->node_projected_ip_vector_);

  Teuchos::RCP< Thyra::MultiVectorBase<ST> >
    b = Thyra::createMultiVector(this->source_load_vector_);

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
  A->apply(Thyra::NOTRANS, *x, y.ptr(), 1.0, 0.0);

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
  node_data->saveNodalDataState(this->node_projected_ip_vector_);
}

} // namespace LCM
