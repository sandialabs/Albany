//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_config.h"

#include <Teuchos_AbstractFactoryStd.hpp>
#include <Teuchos_TestForException.hpp>
#include <fstream>

#ifdef ALBANY_IFPACK2
#include <Thyra_Ifpack2PreconditionerFactory.hpp>
#endif

#include <Phalanx_DataLayout_MDALayout.hpp>

#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_FunctionSpaceTools.hpp>
#include <Shards_CellTopology.hpp>

#include "Adapt_NodalDataVector.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"

namespace LCM {

class ProjectIPtoNodalFieldManager : public Adapt::NodalDataBase::Manager
{
 public:
  // Declare a class hierarchy of mass matrix types. mass_matrix has to be in
  // this specialization, at least for now, because its implementation of fill()
  // is valid only for AlbanyTraits::Residual. Later we might move it up to the
  // nonspecialized class and create separate fill() impls for each trait.
  class MassMatrix;
  class FullMassMatrix;
  class LumpedMassMatrix;
  Teuchos::RCP<MassMatrix>         mass_matrix;
  Teuchos::RCP<Tpetra_MultiVector> ip_field;
  // Start position in the nodal vector database, and number of vectors we're
  // using.
  int ndb_start, ndb_numvecs;

  ProjectIPtoNodalFieldManager() : nwrkr_(0), prectr_(0), postctr_(0) {}

  void
  registerWorker()
  {
    ++nwrkr_;
  }
  int
  nWorker() const
  {
    return nwrkr_;
  }

  void
  initCounters()
  {
    prectr_ = postctr_ = 0;
  }
  int
  incrPreCounter()
  {
    return ++prectr_;
  }
  int
  incrPostCounter()
  {
    return ++postctr_;
  }

 private:
  int nwrkr_, prectr_, postctr_;
};

typedef Intrepid2::Basis<PHX::Device, RealType, RealType> Intrepid2Basis;

class ProjectIPtoNodalFieldQuadrature
{
  typedef PHAL::AlbanyTraits::Residual::MeshScalarT      MeshScalarT;
  PHX::MDField<RealType, Cell, Node, QuadPoint>          bf_;
  PHX::MDField<const RealType, Cell, Node, QuadPoint>    bf_const_;
  PHX::MDField<MeshScalarT, Cell, Node, QuadPoint>       wbf_;
  PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint> wbf_const_;

  Teuchos::RCP<Intrepid2Basis>               intrepid_basis_;
  CellTopologyData                           ctd_;
  Teuchos::RCP<shards::CellTopology>         cell_topo_;
  Kokkos::DynRankView<RealType, PHX::Device> ref_points_, ref_weights_;

 public:
  ProjectIPtoNodalFieldQuadrature(
      Teuchos::ParameterList&              p,
      const Teuchos::RCP<Albany::Layouts>& dl,
      const CellTopologyData&              ctd,
      const int                            degree);
  void
  evaluateBasis(
      const PHX::MDField<const MeshScalarT, Cell, Vertex, Dim>& coords_verts);
  const PHX::MDField<RealType, Cell, Node, QuadPoint>&
  bf() const
  {
    return bf_;
  }
  const PHX::MDField<const RealType, Cell, Node, QuadPoint>&
  bf_const() const
  {
    return bf_const_;
  }
  const PHX::MDField<MeshScalarT, Cell, Node, QuadPoint>&
  wbf() const
  {
    return wbf_;
  }
  const PHX::MDField<const MeshScalarT, Cell, Node, QuadPoint>&
  wbf_const() const
  {
    return wbf_const_;
  }
};

ProjectIPtoNodalFieldQuadrature::ProjectIPtoNodalFieldQuadrature(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl,
    const CellTopologyData&              ctd,
    const int                            degree)
    : ctd_(ctd)
{
  cell_topo_ = Teuchos::rcp(new shards::CellTopology(&ctd_));
  Intrepid2::DefaultCubatureFactory              cubFactory;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cell_topo_, degree);
  const int nqp = cubature->getNumPoints(), nd = cubature->getDimension();
  ref_points_  = Kokkos::DynRankView<RealType, PHX::Device>("XXX", nqp, nd);
  ref_weights_ = Kokkos::DynRankView<RealType, PHX::Device>("XXX", nqp);
  cubature->getCubature(ref_points_, ref_weights_);

  // Support composite Tet<10> in principle; however, I observe that there
  // appears to be something wrong with the quadrature setup for composite
  // Tet<10>, at least at degree 4 and higher. In particular, a linear function
  // is *not* recovered.
  const Teuchos::RCP<Teuchos::ParameterList>& pfp =
      p.get<Teuchos::RCP<Teuchos::ParameterList>>(
          "Parameters From Problem", Teuchos::null);
  const bool composite =
      pfp.is_null() ? false : pfp->get<bool>("Use Composite Tet 10", false);

  ALBANY_ASSERT(
      !composite,
      "\n Project IP to Nodal Field Response not supported with Composite Tet "
      "10s! \n"
          << "Re-run with Use Composite Tet 10 = false or with IP to Nodal "
             "Field Response. \n");

  intrepid_basis_ = Albany::getIntrepid2Basis(ctd, composite);

  typedef PHX::MDALayout<Cell, Node, QuadPoint> Layout;
  Teuchos::RCP<Layout> node_qp_scalar = Teuchos::rcp(new Layout(
      dl->node_qp_scalar->dimension(0), dl->node_qp_scalar->dimension(1), nqp));
  bf_                                 = decltype(bf_)("my BF", node_qp_scalar);
  bf_const_  = decltype(bf_const_)("my BF", node_qp_scalar);
  wbf_       = decltype(wbf_)("my wBF", node_qp_scalar);
  wbf_const_ = decltype(wbf_const_)("my wBF", node_qp_scalar);
  auto bf_data =
      PHX::KokkosViewFactory<RealType, PHX::Device>::buildView(bf_.fieldTag());
  bf_.setFieldData(bf_data);
  bf_const_.setFieldData(bf_data);
  auto wbf_data = PHX::KokkosViewFactory<MeshScalarT, PHX::Device>::buildView(
      wbf_.fieldTag());
  wbf_.setFieldData(wbf_data);
  wbf_const_.setFieldData(wbf_data);
}

void
ProjectIPtoNodalFieldQuadrature::evaluateBasis(
    const PHX::MDField<const MeshScalarT, Cell, Vertex, Dim>& coord_vert)
{
  using namespace Intrepid2;
  typedef CellTools<PHX::Device> CellTools;
  const int nqp = ref_points_.dimension(0), nd = ref_points_.dimension(1),
            nc = coord_vert.dimension(0), nn = coord_vert.dimension(1);
  Kokkos::DynRankView<RealType, PHX::Device> jacobian("JJJ", nc, nqp, nd, nd),
      jacobian_det("JJJ", nc, nqp), weighted_measure("JJJ", nc, nqp),
      val_ref_points("JJJ", nn, nqp);
  CellTools::setJacobian(
      jacobian, ref_points_, coord_vert.get_view(), *cell_topo_);
  CellTools::setJacobianDet(jacobian_det, jacobian);
  intrepid_basis_->getValues(
      val_ref_points, ref_points_, Intrepid2::OPERATOR_VALUE);
  FunctionSpaceTools<PHX::Device>::computeCellMeasure(
      weighted_measure, jacobian_det, ref_weights_);
  FunctionSpaceTools<PHX::Device>::HGRADtransformVALUE(
      bf_.get_view(), val_ref_points);
  FunctionSpaceTools<PHX::Device>::multiplyMeasure(
      wbf_.get_view(), weighted_measure, bf_const_.get_view());
}

static Teuchos::RCP<ProjectIPtoNodalFieldQuadrature>
initQuadMgr(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl,
    const Albany::MeshSpecsStruct*       mesh_specs)
{
  const int min_quad_deg =
      mesh_specs->ctd.node_count > mesh_specs->ctd.vertex_count ? 4 : 2;
  if (mesh_specs->cubatureDegree >= min_quad_deg) return Teuchos::null;
  return Teuchos::rcp(new ProjectIPtoNodalFieldQuadrature(
      p, dl, mesh_specs->ctd, min_quad_deg));
}

template <typename Traits>
typename ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
    EFieldLayout::Enum
    ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::EFieldLayout::
        fromString(const std::string& str)
{
  if (str == "Scalar")
    return scalar;
  else if (str == "Vector")
    return vector;
  else if (str == "Tensor")
    return tensor;
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameterValue,
        "Field Layout value "
            << str
            << "is invalid; valid values are Scalar, Vector, and Tensor.");
  }
}

namespace {
Teuchos::RCP<Teuchos::ParameterList>
getValidProjectIPtoNodalFieldParameters()
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      rcp(new Teuchos::ParameterList("Valid ProjectIPtoNodalField Params"));

  // Don't validate the solver parameters used in the projection solve; let
  // Stratimikos do it.
  valid_pl->sublist("Solver Options").disableRecursiveValidation();

  valid_pl->set<std::string>("Name", "", "Name of field Evaluator");
  valid_pl->set<int>("Number of Fields", 0);
  for (int i = 0; i < 9; ++i) {
    valid_pl->set<std::string>(
        Albany::strint("IP Field Name", i), "", "IP Field prefix");
    valid_pl->set<std::string>(
        Albany::strint("IP Field Layout", i),
        "",
        "IP Field Layout: Scalar, Vector, or Tensor");
  }
  valid_pl->set<bool>(
      "Output to File",
      true,
      "Whether nodal field info should be output to a file");
  valid_pl->set<std::string>("Mass Matrix Type", "Full", "Full or Lumped");
  valid_pl->set<double>("Solver Tolerance", 1e-12, "Linear solver tolerance");

  return valid_pl;
}

void
setDefaultSolverParameters(Teuchos::ParameterList& pl, const double solver_tol)
{
  pl.set<std::string>("Linear Solver Type", "Belos");

  Teuchos::ParameterList& solver_types = pl.sublist("Linear Solver Types");
  Teuchos::ParameterList& belos_types  = solver_types.sublist("Belos");
  belos_types.set<std::string>("Solver Type", "Block CG");

  Teuchos::ParameterList& solver =
      belos_types.sublist("Solver Types").sublist("Block CG");
  solver.set<int>("Maximum Iterations", 1000);
  solver.set<double>("Convergence Tolerance", solver_tol);

#ifdef ALBANY_IFPACK2
  pl.set<std::string>("Preconditioner Type", "Ifpack2");
  Teuchos::ParameterList& prec_types   = pl.sublist("Preconditioner Types");
  Teuchos::ParameterList& ifpack_types = prec_types.sublist("Ifpack2");

  ifpack_types.set<int>("Overlap", 0);
  // Both of these preconditioners are quite effective on the model
  // problem. I'll have to wait to see other problems before I decided whether:
  // (a) Diagonal is always sufficient; (b) ILU(0) with ovl=0 is good; or (c)
  // something more powerful is required (ILUTP or overlap > 0, say).
  const char* prec_type[] = {"RILUK", "Diagonal"};
  ifpack_types.set<std::string>("Prec Type", prec_type[0]);

  Teuchos::ParameterList& ifpack_settings =
      ifpack_types.sublist("Ifpack2 Settings");
  ifpack_settings.set<int>("fact: iluk level-of-fill", 0);
#else
  pl.set<std::string>("Preconditioner Type", "None");
#endif
}

// Represent the mass matrix type by an enumerated type.
struct EMassMatrixType
{
  enum Enum
  {
    full,
    lumped
  };
  static Enum
  fromString(const std::string& str)
  {
    if (str == "Full")
      return full;
    else if (str == "Lumped")
      return lumped;
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          Teuchos::Exceptions::InvalidParameterValue,
          "Mass Matrix Type value "
              << str << "is invalid; valid values are Full and Lumped.");
    }
  }
};
}  // namespace

class ProjectIPtoNodalFieldManager::MassMatrix
{
 public:
  virtual ~MassMatrix() {}

  virtual void
  fill(
      const PHAL::Workset&                                       workset,
      const PHX::MDField<const RealType, Cell, Node, QuadPoint>& bf,
      const PHX::MDField<const RealType, Cell, Node, QuadPoint>& wbf) = 0;

  Teuchos::RCP<Tpetra_CrsMatrix>&
  matrix()
  {
    return matrix_;
  }

  static MassMatrix*
  create(EMassMatrixType::Enum type);

 protected:
  Teuchos::RCP<Tpetra_CrsMatrix> matrix_;
};

class ProjectIPtoNodalFieldManager::FullMassMatrix
    : public ProjectIPtoNodalFieldManager::MassMatrix
{
 public:
  virtual void
  fill(
      const PHAL::Workset&                                       workset,
      const PHX::MDField<const RealType, Cell, Node, QuadPoint>& bf,
      const PHX::MDField<const RealType, Cell, Node, QuadPoint>& wbf)
  {
    const int  num_nodes = bf.dimension(1), num_pts = bf.dimension(2);
    const bool is_static_graph = this->matrix_->isStaticGraph();
    for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
      for (int rnode = 0; rnode < num_nodes; ++rnode) {
        GO                        global_row = workset.wsElNodeID[cell][rnode];
        Teuchos::Array<Tpetra_GO> cols;
        Teuchos::Array<ST>        vals;

        for (int cnode = 0; cnode < num_nodes; ++cnode) {
          const GO global_col = workset.wsElNodeID[cell][cnode];
          cols.push_back(global_col);

          ST mass_value = 0;
          for (int qp = 0; qp < num_pts; ++qp)
            mass_value += wbf(cell, rnode, qp) * bf(cell, cnode, qp);
          vals.push_back(mass_value);
        }
        if (is_static_graph) {
          const LO ret =
              this->matrix_->sumIntoGlobalValues(global_row, cols, vals);
          TEUCHOS_TEST_FOR_EXCEPTION(
              ret != cols.size(),
              std::logic_error,
              "global_row " << global_row
                            << " of mass matrix is missing elements"
                            << std::endl);
        } else {
          this->matrix_->insertGlobalValues(global_row, cols, vals);
        }
      }
    }
  }
};

class ProjectIPtoNodalFieldManager::LumpedMassMatrix
    : public ProjectIPtoNodalFieldManager::MassMatrix
{
 public:
  virtual void
  fill(
      const PHAL::Workset&                                       workset,
      const PHX::MDField<const RealType, Cell, Node, QuadPoint>& bf,
      const PHX::MDField<const RealType, Cell, Node, QuadPoint>& wbf)
  {
    const int  num_nodes = bf.dimension(1), num_pts = bf.dimension(2);
    const bool is_static_graph = this->matrix_->isStaticGraph();
    for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
      for (int rnode = 0; rnode < num_nodes; ++rnode) {
        const GO global_row = workset.wsElNodeID[cell][rnode];
        const Teuchos::Array<Tpetra_GO> cols(1, global_row);
        double                          diag = 0;
        for (std::size_t qp = 0; qp < num_pts; ++qp) {
          double diag_qp = 0;
          for (int cnode = 0; cnode < num_nodes; ++cnode)
            diag_qp += bf(cell, cnode, qp);
          diag += wbf(cell, rnode, qp) * diag_qp;
        }
        const Teuchos::Array<ST> vals(1, diag);
        if (is_static_graph)
          this->matrix_->sumIntoGlobalValues(global_row, cols, vals);
        else
          this->matrix_->insertGlobalValues(global_row, cols, vals);
      }
    }
  }
};

typename ProjectIPtoNodalFieldManager::MassMatrix*
ProjectIPtoNodalFieldManager::MassMatrix::create(EMassMatrixType::Enum type)
{
  switch (type) {
    case EMassMatrixType::full: return new FullMassMatrix();
    case EMassMatrixType::lumped: return new LumpedMassMatrix();
  }
  return nullptr;
}

template <typename Traits>
bool
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::initManager(
    Teuchos::ParameterList* const pl,
    const std::string&            key_suffix)
{
  const std::string key = "ProjectIPtoNodalField_" + key_suffix;
  Teuchos::RCP<Adapt::NodalDataBase> ndb = p_state_mgr_->getNodalDataBase();
  const bool                         isr = ndb->isManagerRegistered(key);
  if (isr)
    mgr_ = Teuchos::rcp_dynamic_cast<ProjectIPtoNodalFieldManager>(
        ndb->getManager(key));
  else {
    EMassMatrixType::Enum mass_matrix_type;
    const std::string& mmstr = pl->get<std::string>("Mass Matrix Type", "Full");
    try {
      mass_matrix_type = EMassMatrixType::fromString(mmstr);
    } catch (const Teuchos::Exceptions::InvalidParameterValue& e) {
      *Teuchos::VerboseObjectBase::getDefaultOStream()
          << "Warning: Mass Matrix Type was set to " << mmstr
          << ", which is invalid; setting to Full." << std::endl;
      mass_matrix_type = EMassMatrixType::full;
    }
    mgr_              = Teuchos::rcp(new ProjectIPtoNodalFieldManager());
    mgr_->mass_matrix = Teuchos::rcp(
        ProjectIPtoNodalFieldManager::MassMatrix::create(mass_matrix_type));
    // Find out our starting position in the nodal database.
    mgr_->ndb_start =
        p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getVecsize();
    ndb->registerManager(key, mgr_);
  }
  mgr_->registerWorker();
  return !isr;
}

template <typename Traits>
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
    ProjectIPtoNodalField(
        Teuchos::ParameterList&              p,
        const Teuchos::RCP<Albany::Layouts>& dl,
        const Albany::MeshSpecsStruct*       mesh_specs)
    : ProjectIPtoNodalFieldBase<PHAL::AlbanyTraits::Residual, Traits>(dl),
      wBF(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
      BF(p.get<std::string>("BF Name"), dl->node_qp_scalar),
#ifdef PROJ_INTERP_TEST
      coords_qp_(p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient),
#endif
      coords_verts_(
          p.get<std::string>("Coordinate Vector Name"),
          dl->vertices_vector)
{
  // todo Some of this doesn't need to be done for all RFs in the case of
  // multiple EBs.

  this->addDependentField(wBF);
  this->addDependentField(BF);
#ifdef PROJ_INTERP_TEST
  this->addDependentField(coords_qp_);
#endif
  this->addDependentField(coords_verts_);
  this->setName("ProjectIPtoNodalField<Residual>");

  quad_mgr_ = initQuadMgr(p, dl, mesh_specs);

  // Get and validate ProjectIPtoNodalField parameter list.
  Teuchos::ParameterList* plist =
      p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist =
      getValidProjectIPtoNodalFieldParameters();
  plist->validateParameters(*reflist, 0);

  output_to_exodus_ = plist->get<bool>("Output to File", true);

  // Number of quad points per cell and dimension.
  const Teuchos::RCP<PHX::DataLayout>& vector_dl      = dl->qp_vector;
  const Teuchos::RCP<PHX::DataLayout>& node_dl        = dl->node_qp_vector;
  const Teuchos::RCP<PHX::DataLayout>& vert_vector_dl = dl->vertices_vector;
  num_pts_                                            = vector_dl->dimension(1);
  num_dims_                                           = vector_dl->dimension(2);
  num_nodes_                                          = node_dl->dimension(1);

  // Number of Fields is read from the input file; this is the number of named
  // fields (scalar, vector, or tensor) to transfer.
  num_fields_ = plist->get<int>("Number of Fields", 0);

  // Surface element prefix, if any.
  const std::string field_name_prefix =
      mesh_specs->ebName == "Surface Element" ? "surf_" : "";

  p_state_mgr_ = p.get<Albany::StateManager*>("State Manager Ptr");

  const std::string key_suffix =
      field_name_prefix +
      (num_fields_ > 0 ?
           plist->get<std::string>(Albany::strint("IP Field Name", 0)) :
           "");
  const bool first = initManager(plist, key_suffix);

  // Resize field vectors.
  ip_field_names_.resize(num_fields_);
  ip_field_layouts_.resize(num_fields_);
  nodal_field_names_.resize(num_fields_);
  ip_fields_.resize(num_fields_);

  for (int field = 0; field < num_fields_; ++field) {
    ip_field_names_[field] =
        field_name_prefix +
        plist->get<std::string>(Albany::strint("IP Field Name", field));
    nodal_field_names_[field] = "proj_nodal_" + ip_field_names_[field];
    ip_field_layouts_[field]  = EFieldLayout::fromString(
        plist->get<std::string>(Albany::strint("IP Field Layout", field)));

    Teuchos::RCP<PHX::DataLayout> qp_layout, node_node_layout;
    switch (ip_field_layouts_[field]) {
      case EFieldLayout::scalar:
        qp_layout        = dl->qp_scalar;
        node_node_layout = dl->node_node_scalar;
        break;
      case EFieldLayout::vector:
        qp_layout        = dl->qp_vector;
        node_node_layout = dl->node_node_vector;
        break;
      case EFieldLayout::tensor:
        qp_layout        = dl->qp_tensor;
        node_node_layout = dl->node_node_tensor;
        break;
    }
    ip_fields_[field] =
        PHX::MDField<const ScalarT>(ip_field_names_[field], qp_layout);
    // Incoming integration point field to transfer.
    this->addDependentField(ip_fields_[field].fieldTag());
    this->p_state_mgr_->registerNodalVectorStateVariable(
        nodal_field_names_[field],
        node_node_layout,
        dl->dummy,
        "all",
        "scalar",
        0.0,
        false,
        output_to_exodus_);
  }

#ifdef PROJ_INTERP_TEST
  ip_field_names_.push_back("linear");
  nodal_field_names_.push_back("proj_nodal_linear");
  ip_field_layouts_.push_back(EFieldLayout::scalar);
  test_ip_field_ =
      decltype(test_ip_field_)(ip_field_names_.back(), dl->qp_scalar);
  ip_fields_.push_back(
      PHX::MDField<const ScalarT>(ip_field_names_.back(), dl->qp_scalar));
  auto&                                                f = ip_fields_.back();
  typedef PHX::KokkosViewFactory<ScalarT, PHX::Device> ViewFactory;
  std::vector<PHX::index_size_type>                    dims;
  dims.push_back(100);
  auto test_data = ViewFactory::buildView(f.fieldTag(), dims);
  f.setFieldData(test_data);
  test_ip_field_.setFieldData(test_data);
  p_state_mgr_->registerNodalVectorStateVariable(
      nodal_field_names_.back(),
      dl->node_node_scalar,
      dl->dummy,
      "all",
      "scalar",
      0.0,
      false,
      output_to_exodus_);
#endif

  if (first)
    mgr_->ndb_numvecs =
        p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getVecsize() -
        mgr_->ndb_start;

// Set up linear solver.
#ifdef ALBANY_IFPACK2
  {
    typedef Thyra::PreconditionerFactoryBase<ST>                  Base;
    typedef Thyra::Ifpack2PreconditionerFactory<Tpetra_CrsMatrix> Impl;

    linearSolverBuilder_.setPreconditioningStrategyFactory(
        Teuchos::abstractFactoryStd<Base, Impl>(), "Ifpack2");
  }
#endif  // IFPACK2

  Teuchos::ParameterList* upl =
      p.get<Teuchos::ParameterList*>("Parameter List");
  const double solver_tol = upl->isType<double>("Solver Tolerance") ?
                                upl->get<double>("Solver Tolerance") :
                                1e-12;
  {  // Send parameters to the solver.
    Teuchos::RCP<Teuchos::ParameterList> solver_list =
        Teuchos::rcp(new Teuchos::ParameterList);
    // Use what has been provided.
    if (plist->isSublist("Solver Options"))
      solver_list->setParameters(plist->sublist("Solver Options"));
    {  // Set the rest of the parameters to their default values.
      Teuchos::ParameterList pl;
      setDefaultSolverParameters(pl, solver_tol);
      solver_list->setParametersNotAlreadySet(pl);
    }
    linearSolverBuilder_.setParameterList(solver_list);
  }

  this->lowsFactory_ = createLinearSolveStrategy(this->linearSolverBuilder_);
  this->lowsFactory_->setVerbLevel(Teuchos::VERB_LOW);
}

template <typename Traits>
void
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::
    postRegistrationSetup(
        typename Traits::SetupData d,
        PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(BF, fm);
  this->utils.setFieldData(wBF, fm);
  for (int field = 0; field < num_fields_; ++field)
    this->utils.setFieldData(ip_fields_[field], fm);
#ifdef PROJ_INTERP_TEST
  this->utils.setFieldData(coords_qp_, fm);
#endif
  this->utils.setFieldData(coords_verts_, fm);
}

template <typename Traits>
void
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::preEvaluate(
    typename Traits::PreEvalData workset)
{
  const int  ctr      = mgr_->incrPreCounter();
  const bool am_first = ctr == 1;
  if (!am_first) return;

  // Reallocate the mass matrix for assembly. Since the matrix is overwritten by
  // a version used for linear algebra having a nonoverlapping row map, we can't
  // just resumeFill. ip_field also alternates between overlapping
  // and nonoverlapping maps and so must be reallocated.
  Teuchos::RCP<const Tpetra_CrsGraph> current_graph =
      p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->getNodalGraph();
  if (Teuchos::nonnull(current_graph)) {
    // Use a graph if it's available.
    mgr_->mass_matrix->matrix() =
        Teuchos::rcp(new Tpetra_CrsMatrix(current_graph));
  } else {
    // Otherwise, construct the graph on the fly.
    const Teuchos::RCP<const Tpetra_Map> ovl_map =
        (p_state_mgr_->getStateInfoStruct()
             ->getNodalDataBase()
             ->getNodalDataVector()
             ->getOverlapMap());
    // Enough for first-order hex, but only a hint.
    const size_t max_num_entries = 27;
    mgr_->mass_matrix->matrix() =
        Teuchos::rcp(new Tpetra_CrsMatrix(ovl_map, ovl_map, max_num_entries));
  }
  mgr_->ip_field = Teuchos::rcp(new Tpetra_MultiVector(
      mgr_->mass_matrix->matrix()->getRowMap(), mgr_->ndb_numvecs, true));
}

template <typename Traits>
void
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::fillRHS(
    const typename Traits::EvalData workset)
{
  Teuchos::RCP<Adapt::NodalDataVector> node_data =
      p_state_mgr_->getStateInfoStruct()
          ->getNodalDataBase()
          ->getNodalDataVector();
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>& wsElNodeID =
      workset.wsElNodeID;

  const int num_fields = num_fields_
#ifdef PROJ_INTERP_TEST
                         + 1
#endif
      ;
  for (int field = 0; field < num_fields; ++field) {
    int node_var_offset, node_var_ndofs;
    node_data->getNDofsAndOffset(
        nodal_field_names_[field], node_var_offset, node_var_ndofs);
    node_var_offset -= mgr_->ndb_start;
    for (unsigned int cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t node = 0; node < num_nodes_; ++node) {
        const GO global_row = wsElNodeID[cell][node];
        for (std::size_t qp = 0; qp < num_pts_; ++qp) {
          switch (ip_field_layouts_[field]) {
            case EFieldLayout::scalar:
              mgr_->ip_field->sumIntoGlobalValue(
                  global_row,
                  node_var_offset,
                  ip_fields_[field](cell, qp) * wBF(cell, node, qp));
              break;
            case EFieldLayout::vector:
              for (std::size_t dim0 = 0; dim0 < num_dims_; ++dim0)
                mgr_->ip_field->sumIntoGlobalValue(
                    global_row,
                    node_var_offset + dim0,
                    (ip_fields_[field](cell, qp, dim0) * wBF(cell, node, qp)));
              break;
            case EFieldLayout::tensor:
              for (std::size_t dim0 = 0; dim0 < num_dims_; ++dim0)
                for (std::size_t dim1 = 0; dim1 < num_dims_; ++dim1)
                  mgr_->ip_field->sumIntoGlobalValue(
                      global_row,
                      node_var_offset + dim0 * num_dims_ + dim1,
                      (ip_fields_[field](cell, qp, dim0, dim1) *
                       wBF(cell, node, qp)));
              break;
          }
        }
      }
    }  // cell
  }    // field
}

#ifdef PROJ_INTERP_TEST
// For Tet<10>, the rhs needs to be integrated with >= degree-3 quadrature to
// recover a linear function exactly.
static double
test_fn(const double x, const double y, const double z)
{
  return 1.5 * x - 0.8 * y + 1.1 * z;
}
#endif

template <typename Traits>
void
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  if (Teuchos::nonnull(quad_mgr_)) {
    quad_mgr_->evaluateBasis(coords_verts_);
    mgr_->mass_matrix->fill(
        workset, quad_mgr_->bf_const(), quad_mgr_->wbf_const());
  } else
    mgr_->mass_matrix->fill(workset, BF, wBF);
#ifdef PROJ_INTERP_TEST
  for (unsigned int cell = 0; cell < workset.numCells; ++cell)
    for (std::size_t qp = 0; qp < num_pts_; ++qp)
      test_ip_field_(cell, qp) = test_fn(
          coords_qp_(cell, qp, 0),
          coords_qp_(cell, qp, 1),
          coords_qp_(cell, qp, 2));
#endif
  fillRHS(workset);
}

template <typename Traits>
void
ProjectIPtoNodalField<PHAL::AlbanyTraits::Residual, Traits>::postEvaluate(
    typename Traits::PostEvalData workset)
{
  const int  ctr     = mgr_->incrPostCounter();
  const bool am_last = ctr == mgr_->nWorker();
  if (!am_last) return;
  mgr_->initCounters();

  typedef Teuchos::ScalarTraits<ST>::magnitudeType MT;
  const ST one = Teuchos::ScalarTraits<ST>::one();

  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();

  mgr_->mass_matrix->matrix()->fillComplete();

  // Right now, ip_field and mass_matrix->matrix() have the same overlapping
  // (row) map.
  //   1. If we're not using a preconditioner, then we could fillComplete the
  // mass matrix with valid 1-1 domain and range maps, export ip_field to b,
  // where b has the mass matrix's range map, and proceed. The linear algebra
  // using the matrix would be limited to matrix-vector products, which would
  // use these valid range and domain maps.
  //   2. However, we want to use Ifpack2, and Ifpack2 assumes the row map is
  // nonoverlapping. (This assumption makes sense because of the type of
  // operations Ifpack2 performs.) Hence I export mass matrix to a new matrix
  // having nonoverlapping row and col maps. As in case 1, I also have to create
  // a compatible b.
  {
    // Get overlapping and nonoverlapping maps.
    const Teuchos::RCP<const Tpetra_CrsMatrix>& mm_ovl =
        mgr_->mass_matrix->matrix();
    if (!mm_ovl->isStaticGraph()) {
      // If this matrix was constructed without a graph, grab the graph now and
      // store it for possible reuse later.
      p_state_mgr_->getStateInfoStruct()->getNodalDataBase()->updateNodalGraph(
          mm_ovl->getCrsGraph());
    }
    const Teuchos::RCP<const Tpetra_Map> ovl_map = mm_ovl->getRowMap();
    const Teuchos::RCP<const Tpetra_Map> map = Tpetra::createOneToOne(ovl_map);
    // Export the mass matrix.
    Teuchos::RCP<Tpetra_Export> e =
        Teuchos::rcp(new Tpetra_Export(ovl_map, map));
    Teuchos::RCP<Tpetra_CrsMatrix> mm =
        rcp(new Tpetra_CrsMatrix(map, mm_ovl->getGlobalMaxNumRowEntries()));
    mm->doExport(*mm_ovl, *e, Tpetra::ADD);
    mm->fillComplete();
    // We don't need the assemble form of the mass matrix any longer.
    mgr_->mass_matrix->matrix() = mm;
    // Now export ip_field.
    Teuchos::RCP<Tpetra_MultiVector> ipf = rcp(new Tpetra_MultiVector(
        mm->getRangeMap(), mgr_->ip_field->getNumVectors()));
    ipf->doExport(*mgr_->ip_field, *e, Tpetra::ADD);
    // Don't need the assemble form of the ip_field either.
    mgr_->ip_field = ipf;
  }
  // Create x in A x = b.
  Teuchos::RCP<Tpetra_MultiVector> node_projected_ip_field =
      rcp(new Tpetra_MultiVector(
          mgr_->mass_matrix->matrix()->getDomainMap(),
          mgr_->ip_field->getNumVectors()));
  const Teuchos::RCP<Tpetra_Operator> tpetra_A = mgr_->mass_matrix->matrix();
  const Teuchos::RCP<Thyra::LinearOpBase<ST>> A =
      Thyra::createLinearOp(tpetra_A);
  Teuchos::RCP<Thyra::LinearOpWithSolveBase<ST>> nsA = lowsFactory_->createOp();
  Thyra::initializeOp<ST>(*lowsFactory_, A, nsA.ptr());
  Teuchos::RCP<Thyra::MultiVectorBase<ST>>
      x = Thyra::createMultiVector<ST, LO, Tpetra_GO, KokkosNode>(
          node_projected_ip_field),
      b = Thyra::createMultiVector<ST, LO, Tpetra_GO, KokkosNode>(
          mgr_->ip_field);

  // Compute the column norms of the right-hand side b. If b = 0, no need to
  // proceed.
  Teuchos::Array<MT> norm_b(mgr_->ip_field->getNumVectors());
  Thyra::norms_2(*b, norm_b());
  bool b_is_zero = true;
  for (int i = 0; i < mgr_->ip_field->getNumVectors(); ++i)
    if (norm_b[i] != 0) {
      b_is_zero = false;
      break;
    }
  if (b_is_zero) return;

  Thyra::SolveStatus<ST> solveStatus =
      Thyra::solve(*nsA, Thyra::NOTRANS, *b, x.ptr());
#ifdef ALBANY_DEBUG
  *out << "\nBelos LOWS Status: " << solveStatus << std::endl;

  // Compute residual and ST check convergence.
  Teuchos::RCP<Thyra::MultiVectorBase<ST>> y =
      Thyra::createMembers(x->range(), x->domain());

  // Compute y = A*x, where x is the solution from the linear solver.
  A->apply(Thyra::NOTRANS, *x, y.ptr(), 1.0, 0.0);

  // Compute A*x - b = y - b.
  Thyra::update(-one, *b, y.ptr());
  Teuchos::Array<MT> norm_res(mgr_->ip_field->getNumVectors());
  Thyra::norms_2(*y, norm_res());
  // Print out the final relative residual norms.
  *out << "Final relative residual norms" << std::endl;
  for (int i = 0; i < mgr_->ip_field->getNumVectors(); ++i) {
    const double rel_res = norm_res[i] == 0 ? 0 : norm_res[i] / norm_b[i];
    *out << "RHS " << i + 1 << " : " << std::setw(16) << std::right << rel_res
         << std::endl;
  }
#endif
  {  // Store the overlapped vector data back in stk.
    const Teuchos::RCP<const Tpetra_Map> ovl_map =
                                             (p_state_mgr_->getStateInfoStruct()
                                                  ->getNodalDataBase()
                                                  ->getNodalDataVector()
                                                  ->getOverlapMap()),
                                         map =
                                             node_projected_ip_field->getMap();
    Teuchos::RCP<Tpetra_MultiVector> npif = rcp(new Tpetra_MultiVector(
        ovl_map, node_projected_ip_field->getNumVectors()));
    Teuchos::RCP<Tpetra_Import>      im =
        Teuchos::rcp(new Tpetra_Import(map, ovl_map));
    npif->doImport(*node_projected_ip_field, *im, Tpetra::ADD);
    p_state_mgr_->getStateInfoStruct()
        ->getNodalDataBase()
        ->getNodalDataVector()
        ->saveNodalDataState(npif, mgr_->ndb_start);
  }
}

}  // namespace LCM
