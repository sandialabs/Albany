//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iomanip>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <ma.h>
#include <PCU.h>
#include <parma.h>
#include <apfZoltan.h>
#include <apfMDS.h> // for createMdsMesh

#include "AAdapt_UnifSizeField.hpp"
#include "AAdapt_UnifRefSizeField.hpp"
#include "AAdapt_NonUnifRefSizeField.hpp"
#include "AAdapt_AlbanySizeField.hpp"
#ifdef SCOREC_SPR
#include "AAdapt_SPRSizeField.hpp"
#endif

#include "AAdapt_RC_Manager.hpp"

#include "AAdapt_MeshAdapt.hpp"

AAdapt::MeshAdapt::
MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
          const Teuchos::RCP<ParamLib>& paramLib_,
          const Albany::StateManager& StateMgr_,
          const Teuchos::RCP<AAdapt::rc::Manager>& refConfigMgr_,
          const Teuchos::RCP<const Teuchos_Comm>& commT_)
  : AbstractAdapterT(params_, paramLib_, StateMgr_, commT_),
    remeshFileIndex(1), rc_mgr(refConfigMgr_)
{
  disc = StateMgr_.getDiscretization();

  pumi_discretization = Teuchos::rcp_dynamic_cast<Albany::PUMIDiscretization>(disc);

  Teuchos::RCP<Albany::PUMIMeshStruct> pumiMeshStruct =
    pumi_discretization->getPUMIMeshStruct();

  mesh = pumiMeshStruct->getMesh();

  const std::string& method = params_->get("Method", "");
  if (method == "RPI Unif Size")
    szField = Teuchos::rcp(new AAdapt::UnifSizeField(pumi_discretization));
  else if (method == "RPI UnifRef Size")
    szField = Teuchos::rcp(new AAdapt::UnifRefSizeField(pumi_discretization));
  else if (method == "RPI NonUnifRef Size")
    szField = Teuchos::rcp(new AAdapt::NonUnifRefSizeField(pumi_discretization));
  else if (method == "RPI Albany Size")
    szField = Teuchos::rcp(new AAdapt::AlbanySizeField(pumi_discretization));
#ifdef SCOREC_SPR
  else if (method == "RPI SPR Size")
    szField = Teuchos::rcp(new AAdapt::SPRSizeField(pumi_discretization));
#endif
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "should not be here");

  // Save the initial output file name
  base_exo_filename = pumiMeshStruct->outputFileName;

  adaptation_method = params_->get<std::string>("Method");

  if ( adaptation_method.compare(0,15,"RPI SPR Size") == 0 )
    checkValidStateVariable(StateMgr_,params_->get<std::string>("State Variable",""));

  initRcMgr();
}

namespace {
inline int getValueType (const PHX::DataLayout& dl) {
  switch (dl.rank() - 2) {
  case 0: return apf::SCALAR;
  case 1: return apf::VECTOR;
  case 2: return apf::MATRIX;
  default:
    std::stringstream ss;
    ss << "not a valid rank: " << dl.rank() - 2;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, ss.str());
    return -1;
  }
}
} // namespace

void AAdapt::MeshAdapt::initRcMgr () {
  if (rc_mgr.is_null()) return;
  Teuchos::RCP<Albany::PUMIMeshStruct> pumiMeshStruct =
    pumi_discretization->getPUMIMeshStruct();
  // A field to store the reference configuration x (displacement). At each
  // adapatation, it will be interpolated to the new mesh.
  //rc-todo Always apf::VECTOR?
  //rc-todo Generalize to Albany::AbstractDiscretization.
  pumiMeshStruct->createNodalField("x_accum", apf::VECTOR);
  if (rc_mgr->usingProjection()) {
    for (rc::Manager::Field::iterator it = rc_mgr->fieldsBegin(),
           end = rc_mgr->fieldsEnd();
         it != end; ++it) {
      const int value_type = getValueType(*(*it)->layout);
      for (int i = 0; i < (*it)->num_g_fields; ++i) {
        pumiMeshStruct->createNodalField(
          (*it)->get_g_name(i).c_str(), value_type);
      }
    }
    rc_mgr->initProjector(pumi_discretization->getNodeMapT(),
                          pumi_discretization->getOverlapNodeMapT());
  }
#ifdef AMBDEBUG
  // Create a field that never changes. It's interp'ed from now mesh to the
  // next. In the initial configuration, each component is simply a
  // coordinate value.
  const int dim = pumi_discretization->getNumDim();
  pumiMeshStruct->createNodalField("test_interp_field", apf::VECTOR);
  const Teuchos::ArrayRCP<const double>&
    coords = pumi_discretization->getCoordinates();
  Teuchos::Array<double> f(coords.size());
  memcpy(&f[0], &coords[0], coords.size()*sizeof(double));
  pumi_discretization->setField("test_interp_field", &f[0], true, 0, dim);
#endif
}

AAdapt::MeshAdapt::~MeshAdapt() {}

bool AAdapt::MeshAdapt::queryAdaptationCriteria(int iter)
{
  adapt_params_->set<int>("LastIter", iter);

  std::string strategy = adapt_params_->get<std::string>("Remesh Strategy", "None");

  if (strategy.compare("Continuous") == 0)
    return iter > 1;

  if (strategy.compare("PLDriven") == 0){
    if(adapt_params_->get<bool>("AdaptNow", false)){

      adapt_params_->set<bool>("AdaptNow", false);
      return iter > 1;

    }
    return false;
  }

  Teuchos::Array<int> remesh_iter = adapt_params_->get<Teuchos::Array<int> >("Remesh Step Number");
  for (int i = 0; i < remesh_iter.size(); i++)
    if (iter == remesh_iter[i])
      return true;
  return false;
}


void AAdapt::MeshAdapt::initAdapt()
{
  if (PCU_Comm_Self() == 0) {
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "Adapting mesh using AAdapt::MeshAdapt method        " << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  }

  // Create a remeshed output file naming convention by adding the
  // remesh_file_index_ ahead of the period
  const std::size_t found = base_exo_filename.find("exo");
  if (found != std::string::npos) {
    std::ostringstream ss;
    std::string str = base_exo_filename;
    ss << "_" << remeshFileIndex << ".";
    str.replace(str.find('.'), 1, ss.str());

    *output_stream_ << "Remeshing: renaming exodus output file to - " << str << std::endl;

    // Open the new exodus file for results
    pumi_discretization->reNameExodusOutput(str);

    remeshFileIndex++;
  }

  // attach qp data to mesh if solution transfer is turned on
  should_transfer_ip_data = adapt_params_->get<bool>("Transfer IP Data", false);
  // If the mesh adapt loop is run, we have to transfer state for SPR.
  if (Teuchos::nonnull(rc_mgr)) should_transfer_ip_data = true;

  szField->setParams(adapt_params_);

}

void AAdapt::MeshAdapt::beforeAdapt()
{
  if (should_transfer_ip_data)
    pumi_discretization->attachQPData();
  szField->copyInputFields();
}

void AAdapt::MeshAdapt::adaptInPartition()
{
  szField->computeError();

  szField->configure(adapt_params_);

  szField->freeSizeField();
}

namespace {
  void setUnitEntWeights(ma::Mesh* m, ma::Tag* weights, int dim)
  {
    ma::Entity* e;
    double one = 1.0;
    apf::MeshIterator* it = m->begin(dim);
    while ((e = m->iterate(it)))
      m->setDoubleTag(e,weights,&one);
    m->end(it);
  }

  void runParmaVtxElm(ma::Mesh* m, double maxImb)
  {
    ma::Tag* weights = m->createDoubleTag("ma_weight",1);
    setUnitEntWeights(m,weights,0);
    setUnitEntWeights(m,weights,m->getDimension());
    apf::Balancer* b = Parma_MakeVtxElmBalancer(m);
    b->balance(weights,maxImb);
    delete b;
    apf::removeTagFromDimension(m,weights,0);
    apf::removeTagFromDimension(m,weights,m->getDimension());
    m->destroyTag(weights);
  }

  void runZoltanBal(ma::Mesh* m, double maxImb)
  {
    ma::Tag* weights = Parma_WeighByMemory(m);
    apf::Balancer* b = makeZoltanBalancer(m, apf::GRAPH, apf::REPARTITION);
    b->balance(weights,maxImb);
    delete b;
    apf::removeTagFromDimension(m,weights,m->getDimension());
    m->destroyTag(weights);
  }

  struct albBalancer {
    double imb;
    std::string method;
    void (*bal)(ma::Mesh* m, double maxImb);
  };

  albBalancer* postBalance(std::string method, double maxImb) {
    albBalancer* b = new albBalancer;
    b->imb = maxImb;
    if(method == std::string("zoltan")) {
      b->bal = runZoltanBal;
    } else if(method == std::string("parma")) {
      b->bal = runParmaVtxElm;
    } else {
      b->bal = NULL;
    }
    return b;
  }
}

void AAdapt::MeshAdapt::afterAdapt()
{
  Teuchos::Array<std::string> defaultStArgs = 
     Teuchos::tuple<std::string>("zoltan", "parma", "parma");
  double maxImb = adapt_params_->get<double>("Maximum LB Imbalance", 1.30);

  albBalancer* b = postBalance(defaultStArgs[2], maxImb);
  b->bal(mesh, b->imb);
  delete b;

  mesh->verify();

  szField->freeInputFields();
  // Throw away all the Albany data structures and re-build them from the mesh
  // Note that the solution transfer for the QP fields happens in this call
  pumi_discretization->updateMesh(should_transfer_ip_data);

  // detach QP fields from the apf mesh
  if (should_transfer_ip_data)
    pumi_discretization->detachQPData();
}

struct AdaptCallbackOf : public Parma_GroupCode
{
  AAdapt::MeshAdapt* adapter;
  void run(int group) {
    adapter->adaptInPartition();
  }
};

/* Adaptation loop.
 *   Namespace al and method adaptMeshLoop implement the following operation. We
 * have current coordinates c and the solution vector of displacements d. Now we
 * need to update the coordinates to be c' = c + d and then hand c' to the
 * SCOREC remesher.
 *   One problem is that if we simply add d to c, some tets may flip from having
 * positive volume to negative volume. This loop implements a stepping procedure
 * that tries out c'' = c + alpha d, with the goal of moving alpha from 0 to
 * 1. It tries alpha = 1 first to take advantage of the easy case of no volume
 * sign flips. If that fails, then it backs off. Once it finds 0 < alpha < 1 for
 * which c'' has all positive-volume tets, c'' is passed to the SCOREC
 * remesher. The remesher also interpolates (1 - alpha) d, which is the
 * displacement remaining to be accumulated into the coordinates, to the new
 * mesh. This procedure repeats until alpha = 1.
 */
namespace al {
void anlzCoords(
  const Teuchos::RCP<const Albany::PUMIDiscretization>& pumi_disc);
void writeMesh(
  const Teuchos::RCP<Albany::PUMIDiscretization>& pumi_disc);
double findAlpha(
  const Teuchos::RCP<Albany::PUMIDiscretization>& pumi_disc,
  const Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr,
  // Max number of iterations to spend if a successful alpha is already found.
  const int n_iterations_if_found,
  // Max number of iterations before failure is declared. Must be >=
  // n_iterations_if_found.
  const int n_iterations_to_fail);

bool correctnessTestSkip () {
#ifndef AMBDEBUG
  return false;
#else
  static int cnt = 0;
  if ( ! amb::Options::get()->params()->isType<int>("nadapt")) return false;
  return ++cnt > amb::Options::get()->params()->get<int>("nadapt");
#endif
}
} // namespace al

void adaptShrunken(apf::Mesh2* m, double min_part_density,
                   Parma_GroupCode& callback);

bool AAdapt::MeshAdapt::adaptMesh(
  const Teuchos::RCP<const Tpetra_Vector>& solution,
  const Teuchos::RCP<const Tpetra_Vector>&)
{
#ifdef AMBDEBUG
  al::anlzCoords(pumi_discretization);
#endif

  AdaptCallbackOf callback;
  callback.adapter = this;
  const double
    min_part_density = adapt_params_->get<double>("Minimum Part Density", 1000);

  initAdapt();

  bool success;
  if (rc_mgr.is_null()) {
    // Old method. No reference configuration updating.
    if ( ! al::correctnessTestSkip()) {
      beforeAdapt();
      adaptShrunken(pumi_discretization->getPUMIMeshStruct()->getMesh(),
                    min_part_density, callback);
      afterAdapt();
    }
    success = true;
  } else {
    success = adaptMeshWithRc(min_part_density, callback);
  }

#ifdef AMBDEBUG
  al::anlzCoords(pumi_discretization);
  al::writeMesh(pumi_discretization);
#endif
  return success;
}

bool AAdapt::MeshAdapt::
adaptMeshWithRc (const double min_part_density, Parma_GroupCode& callback) {
  const bool overlapped = true;

  rc_mgr->beginAdapt();

  if (rc_mgr->usingProjection()) {
    // Give PUMI the nodal data.
    for (rc::Manager::Field::iterator it = rc_mgr->fieldsBegin(),
           end = rc_mgr->fieldsEnd();
         it != end; ++it)
      for (int i = 0; i < (*it)->num_g_fields; ++i) {
        const Teuchos::RCP<Tpetra_MultiVector>&
          mv = rc_mgr->getNodalField(**it, i, overlapped);
        const std::size_t n = mv->getLocalLength();
        const int ncol = mv->getNumVectors();
        Teuchos::Array<double> data(n * ncol);
        // non-interleaved -> interleaved ordering.
        //rcu-todo Figure out these ordering details. What sets the ordering
        // requirements? Can we change by field at runtime?
        for (int c = 0; c < ncol; ++c) {
          Teuchos::ArrayRCP<RealType>
            v = mv->getVectorNonConst(c)->getDataNonConst();
          for (size_t k = 0; k < n; ++k) data[ncol*k + c] = v[k];
        }
        pumi_discretization->setField((*it)->get_g_name(i).c_str(), &data[0],
                                      overlapped, 0, ncol);
      }
  }

  const bool success = adaptMeshLoop(min_part_density, callback);
  rc_mgr->endAdapt(pumi_discretization->getNodeMapT(),
                   pumi_discretization->getOverlapNodeMapT());

  if (rc_mgr->usingProjection()) {
    // Get the nodal data from PUMI.
    for (rc::Manager::Field::iterator it = rc_mgr->fieldsBegin(),
           end = rc_mgr->fieldsEnd();
         it != end; ++it)
      for (int i = 0; i < (*it)->num_g_fields; ++i) {
        const Teuchos::RCP<Tpetra_MultiVector>&
          mv = rc_mgr->getNodalField(**it, i, overlapped);
        const std::size_t n = mv->getLocalLength();
        const int ncol = mv->getNumVectors();
        Teuchos::Array<double> data(n * ncol);
        pumi_discretization->getField((*it)->get_g_name(i).c_str(), &data[0],
                                      overlapped, 0, ncol);
        for (int c = 0; c < ncol; ++c) {
          Teuchos::ArrayRCP<RealType>
            v = mv->getVectorNonConst(c)->getDataNonConst();
          for (size_t k = 0; k < n; ++k) v[k] = data[ncol*k + c];
        }
      }
  }

  return success;
}

// Later: I'm keeping this loop because it's pretty effective at what it's
// intended to do. However, if this loop has to execute more than one iteration,
// it's almost certainly the case that we're already sunk. Iterating more than
// once implies the displacement solution is turning an element inside
// out. That's bad.
bool AAdapt::MeshAdapt::
adaptMeshLoop (const double min_part_density, Parma_GroupCode& callback) {
  const int
    n_max_outer_iterations = 10,
    n_max_inner_iterations_if_found = 20,
    // -log2(eps) = 52. Since this iteration uses bisection, we can go for about
    // 50 iterations before failure is definite. The iteration is cheap relative
    // to other parts of adaptation, so there is no reason not to set this
    // parameter to ~50.
    n_max_inner_iterations_to_fail = 50;
  bool success = false;
  double alpha_accum = 0;
  for (int it = 0; it < n_max_outer_iterations; ++it) {
    const double alpha = al::findAlpha(
      pumi_discretization, rc_mgr, n_max_inner_iterations_if_found,
      n_max_inner_iterations_to_fail);
    if (Teuchos::DefaultComm<int>::getComm()->getRank() == 0) {
      alpha_accum = alpha_accum + alpha*(1 - alpha_accum);
      std::cout << "amb: adaptMeshLoop it " << it << " alpha " << alpha
                << " alpha_accum " << alpha_accum << "\n";
    }
    if (alpha == 0) { success = false; break; }

    if ( ! al::correctnessTestSkip()) {
      beforeAdapt();
      adaptShrunken(pumi_discretization->getPUMIMeshStruct()->getMesh(),
                    min_part_density, callback);
      afterAdapt();
    }

    // Resize x.
    rc_mgr->get_x() = Teuchos::rcp(
      new Tpetra_Vector(pumi_discretization->getSolutionFieldT()->getMap()));
    // Copy ref config data, now interp'ed to new mesh, into it.
    pumi_discretization->getField(
      "x_accum", &rc_mgr->get_x()->get1dViewNonConst()[0], false);
    
    if (alpha == 1) { success = true; break; }
  }

  return success;
}

void AAdapt::MeshAdapt::checkValidStateVariable(
  const Albany::StateManager& state_mgr_,
  const std::string name)
{
  if (name.length() > 0) {
    // does state variable exist?
    std::string stateName;

    Albany::StateArrays& sa = disc->getStateArrays();
    Albany::StateArrayVec& esa = sa.elemStateArrays;
    Teuchos::RCP<Albany::StateInfoStruct> stateInfo = state_mgr_.getStateInfoStruct();
    bool exists = false;
    for(unsigned int i = 0; i < stateInfo->size(); i++) {
      stateName = (*stateInfo)[i]->name;
      if ( name.compare(0,100,stateName) == 0 ) {
        exists = true;
        break;
      }
    }
    if (!exists)
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                 "Error!    Invalid State Variable Parameter!");

    // is state variable a 3x3 tensor?

    std::vector<int> dims;
    esa[0][name].dimensions(dims);
    int size = dims.size();
    if (size != 4)
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                 "Error! Invalid State Variable Parameter \"" << name << "\" looking for \"" << stateName << "\"" << std::endl);
  }
}

Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::MeshAdapt::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    getGenericAdapterParams("ValidMeshAdaptParams");
  Teuchos::Array<int> defaultArgs;
  Teuchos::Array<std::string> defaultStArgs;

  validPL->set<Teuchos::Array<int> >("Remesh Step Number", defaultArgs, "Iteration step at which to remesh the problem");
  validPL->set<std::string>("Remesh Strategy", "", "Strategy to use when remeshing: Continuous - remesh every step.");
  validPL->set<bool>("AdaptNow", false, "Used to force an adaptation step");
  validPL->set<bool>("Should Coarsen", true, "Set to false to disable mesh coarsening operations.");
  validPL->set<int>("Max Number of Mesh Adapt Iterations", 1, "Number of iterations to limit meshadapt to");
  validPL->set<double>("Target Element Size", 0.1, "Seek this element size when isotropically adapting");
  validPL->set<double>("Error Bound", 0.1, "Max relative error for error-based adaptivity");
  validPL->set<std::string>("State Variable", "", "SPR operates on this variable");
  validPL->set<Teuchos::Array<std::string> >("Load Balancing", defaultStArgs, "Turn on predictive load balancing");
  validPL->set<double>("Maximum LB Imbalance", 1.3, "Set maximum imbalance tolerance for predictive laod balancing");
  validPL->set<std::string>("Adaptation Displacement Vector", "", "Name of APF displacement field");
  validPL->set<bool>("Transfer IP Data", false, "Turn on solution transfer of integration point data");
  validPL->set<double>("Minimum Part Density", 1000, "Minimum elements per part: triggers partition shrinking");
  if (Teuchos::nonnull(rc_mgr)) rc_mgr->getValidParameters(validPL);

  return validPL;
}

static double getAveragePartDensity(apf::Mesh* m)
{
  double nElements = m->count(m->getDimension());
  PCU_Add_Doubles(&nElements, 1);
  return nElements / PCU_Comm_Peers();
}

static int getShrinkFactor(apf::Mesh* m, double minPartDensity)
{
  double partDensity = getAveragePartDensity(m);
  int factor = 1;
  while (partDensity < minPartDensity) {
    if (factor >= PCU_Comm_Peers())
      break;
    factor *= 2;
    partDensity *= 2;
  }
  assert(PCU_Comm_Peers() % factor == 0);
  return factor;
}

static void warnAboutShrinking(int factor)
{
  int nprocs = PCU_Comm_Peers() / factor;
  if (!PCU_Comm_Self())
    fprintf(stderr,"sensing mesh is spread too thin: adapting with %d procs\n",
        nprocs);
}

void adaptShrunken(apf::Mesh2* m, double minPartDensity,
                   Parma_GroupCode& callback)
{
  int factor = getShrinkFactor(m, minPartDensity);
  if (factor == 1)
    callback.run(0);
  else {
    warnAboutShrinking(factor);
    Parma_ShrinkPartition(m, factor, callback);
  }
}

// Adaptation loop. Looping is necessary only if updating the coordinates leads
// to negative simplices.
//rc-todo Can I deal with the "APF warning: 2 empty parts" issue by looping, too?
namespace al {
typedef double ExtremumFn (const double, const double);
inline double mymin (const double a, const double b) { return std::min(a, b); }
inline double mymax (const double a, const double b) { return std::max(a, b); }

template<ExtremumFn extremum_fn>
void dispExtremum (
  const Teuchos::ArrayRCP<const double>& x, const int dim,
  const std::string& extremum_str, const Teuchos::EReductionType rt)
{
  double my_vals[3], global_vals[3];
  const std::size_t nx = x.size() / dim;
  for (std::size_t j = 0; j < dim; ++j) my_vals[j] = x[j];
  const double* px = &x[dim];
  for (std::size_t i = 1; i < nx; ++i) {
    for (std::size_t j = 0; j < dim; ++j)
      my_vals[j] = extremum_fn(my_vals[j], px[j]);
    px += dim;
  }
  const Teuchos::RCP<const Teuchos::Comm<int> >
    comm = Teuchos::DefaultComm<int>::getComm();
  Teuchos::reduceAll(*comm, rt, dim, my_vals, global_vals);
  if (comm->getRank() == 0) {
    std::cout << "amb: " << extremum_str << " ";
    for (std::size_t j = 0; j < dim; ++j) std::cout << " " << global_vals[j];
    std::cout << std::endl;
  }
}

// For analysis.
void anlzCoords (
  const Teuchos::RCP<const Albany::PUMIDiscretization>& pumi_disc)
{
  // x = coords + displacement.
  const int dim = pumi_disc->getNumDim();
  const Teuchos::ArrayRCP<const double>& coords = pumi_disc->getCoordinates();
  if (coords.size() == 0) return;
  const Teuchos::RCP<const Tpetra_Vector>
    soln = pumi_disc->getSolutionFieldT(true);
  const Teuchos::ArrayRCP<const ST> soln_data = soln->get1dView();
  const Teuchos::ArrayRCP<double> x(coords.size());
  const int spdim = pumi_disc->getNumDim(), neq = pumi_disc->getNumEq();
  for (std::size_t i = 0, j = 0; i < coords.size(); i += spdim, j += neq)
    for (int k = 0; k < spdim; ++k)
      x[i+k] = coords[i+k] + soln_data[j+k];
  // Display min and max extent in each axis-aligned dimension.
  dispExtremum<mymin>(x, dim, "min", Teuchos::REDUCE_MIN);
  dispExtremum<mymax>(x, dim, "max", Teuchos::REDUCE_MAX);
}

void writeMesh (
  const Teuchos::RCP<Albany::PUMIDiscretization>& pumi_disc)
{
  return;
  static int ncalls = 0;
  std::stringstream ss;
  ss << "mesh_" << ncalls << ".vtk";
  ++ncalls;
  pumi_disc->writeMeshDebug(ss.str());
}

// Helper struct for updateCoordinates. Keep track of data relevant to update
// coordinates and restore the original state in the case of error.
struct CoordState {
private:
  const Teuchos::RCP<Tpetra_Vector> soln_ol_;
  double alpha_;

public:
  Teuchos::ArrayRCP<double> coords;
  const Teuchos::RCP<Tpetra_Vector> soln_nol;
  const Teuchos::ArrayRCP<const ST> soln_ol_data;

  CoordState (
    const Teuchos::RCP<Albany::PUMIDiscretization>& pumi_disc)
    : soln_ol_(pumi_disc->getSolutionFieldT(true)),
      soln_nol(pumi_disc->getSolutionFieldT(false)),
      soln_ol_data(soln_ol_->get1dView())
  {
    coords.deepCopy(pumi_disc->getCoordinates()());
  }

  void set_alpha (const double alpha) {
    alpha_ = alpha;
    soln_ol_->scale(alpha);
    soln_nol->scale(alpha);
  }

  void restore () {
    soln_ol_->scale(1/alpha_);
    soln_nol->scale(1/alpha_);
  }
};

void updateCoordinates (
  const Teuchos::RCP<Albany::PUMIDiscretization>& pumi_disc,
  const CoordState& cs, const Teuchos::ArrayRCP<double>& x)
{
  // Albany::PUMIDiscretization uses interleaved DOF and coordinates, so we
  // can sum coords and soln_data straightforwardly.
  const int spdim = pumi_disc->getNumDim(), neq = pumi_disc->getNumEq();
  for (std::size_t i = 0, j = 0; i < cs.coords.size(); i += spdim, j += neq)
    for (int k = 0; k < spdim; ++k)
      x[i+k] = cs.coords[i+k] + cs.soln_ol_data[j+k];
  pumi_disc->setCoordinates(x);
}

void updateRefConfig (
  const Teuchos::RCP<Albany::PUMIDiscretization>& pumi_disc,
  const Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr,
  const CoordState& cs)
{
  // x_refconfig += displacement (nonoverlapping).
  rc_mgr->update_x(*cs.soln_nol);
  // Save x_refconfig to the mesh database so it is interpolated after mesh
  // adaptation.
  pumi_disc->setField("x_accum", &rc_mgr->get_x()->get1dView()[0], false);
}

void updateSolution (
  const Teuchos::RCP<Albany::PUMIDiscretization>& pumi_disc,
  const double alpha, CoordState& cs)
{
  // Set solution to (1 - alpha) solution.
  //   Undo the scaling the findAlpha loop applied to get back to original:
  // 1/alpha.
  //   Then apply the new scaling, 1 - alpha, to set the remaining amount of
  // displacement for which the adaptation has to account.
  if (alpha == 1) {
    // Probably a little faster to do this in this end-member case.
    cs.soln_nol->putScalar(0);
  } else cs.soln_nol->scale((1 - alpha)/alpha);
  // The -1 is a dummy value for time, which doesn't actually get used in this
  // operation.
  pumi_disc->writeSolutionToMeshDatabaseT(*cs.soln_nol, -1, false);
}

// Find 0 < alpha < 1 by bisection such that c = coords + alpha displacement
// (where displacement is given by the solution field) has no negative
// simplices. If this function succeeds, the coordinates are updated to c, the
// reference configuration is updated, and the solution field is set to (1 -
// alpha) [original solution].
double findAlpha (
  const Teuchos::RCP<Albany::PUMIDiscretization>& pumi_disc,
  const Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr,
  // Max number of iterations to spend if a successful alpha is already found.
  const int n_iterations_if_found,
  // Max number of iterations before failure is declared. Must be >=
  // n_iterations_if_found.
  const int n_iterations_to_fail)
{
  CoordState cs(pumi_disc);
  
  // Temp storage for proposed new coordinates.
  const Teuchos::ArrayRCP<double> x(cs.coords.size());

  double alpha_lo = 0, alpha_hi = 1, alpha = 1;
  for (int it = 0 ;; ) {
    cs.set_alpha(alpha);
    updateCoordinates(pumi_disc, cs, x);
#ifdef AMBDEBUG
    if (it == 0) al::writeMesh(pumi_disc);
#endif

    ++it;
    const long n_negative_simplices = apf::verifyVolumes(
      pumi_disc->getPUMIMeshStruct()->getMesh(), false);
    // Adjust alpha bounds.
    if (n_negative_simplices == 0)
      alpha_lo = alpha;
    else
      alpha_hi = alpha;

    if (Teuchos::DefaultComm<int>::getComm()->getRank() == 0) {
      static const int w = 8;
      std::cout.precision(4);
      std::cout << "amb: findAlpha it " << std::setw(2) << it
                << " negative volumes " << std::setw(4)
                << n_negative_simplices << " alpha " << std::setw(w) << alpha
                << " in [" << std::setw(w) << alpha_lo << ", " << std::setw(w)
                << alpha_hi << "]\n";
    }

    // Perfect (and typical) case: success on first try, and made it all the way
    // to alpha == 1.
    if (n_negative_simplices == 0 && alpha_lo == alpha_hi) break;
    // Don't waste effort; we have a decent alpha.
    if (it >= n_iterations_if_found && alpha_lo > 0) {
      cs.restore();
      cs.set_alpha(alpha_lo);
      updateCoordinates(pumi_disc, cs, x);
      break;
    }
    // alpha_lo == 0 and we're out of iterations.
    if (it == n_iterations_to_fail) break;

    alpha = 0.5*(alpha_lo + alpha_hi);
    // Restore original for next try.
    cs.restore();
  }

  if (alpha_lo > 0) {
    // The coordinates are already set according to alpha.
    updateRefConfig(pumi_disc, rc_mgr, cs);
    updateSolution(pumi_disc, alpha_lo, cs);
  }

  return alpha_lo;
}
} // namespace al
