//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_MeshAdapt.hpp"
#include "AAdapt_MeshAdapt_Def.hpp"

MESHADAPT_INSTANTIATE_TEMPLATE_CLASS(AAdapt::MeshAdaptT)

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

namespace apf { extern long verifyVolumes(apf::Mesh* m); }

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
  const Teuchos::RCP<const AlbPUMI::AbstractPUMIDiscretization>& pumi_disc)
{
  // x = coords + displacement.
  const int dim = pumi_disc->getNumDim();
  const Teuchos::ArrayRCP<const double>& coords = pumi_disc->getCoordinates();
  if (coords.size() == 0) return;
  const Teuchos::RCP<const Tpetra_Vector>
    soln = pumi_disc->getSolutionFieldT(true);
  const Teuchos::ArrayRCP<const ST> soln_data = soln->get1dView();
  const Teuchos::ArrayRCP<double> x(coords.size());
  for (std::size_t i = 0; i < coords.size(); ++i)
    x[i] = coords[i] + soln_data[i];
  // Display min and max extent in each axis-aligned dimension.
  dispExtremum<mymin>(x, dim, "min", Teuchos::REDUCE_MIN);
  dispExtremum<mymax>(x, dim, "max", Teuchos::REDUCE_MAX);
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
    const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& pumi_disc)
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
  const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& pumi_disc,
  const CoordState& cs, const Teuchos::ArrayRCP<double>& x)
{
  // AlbPUMI::FMDBDiscretization uses interleaved DOF and coordinates, so we
  // can sum coords and soln_data straightforwardly.
  for (std::size_t i = 0; i < cs.coords.size(); ++i)
    x[i] = cs.coords[i] + cs.soln_ol_data[i];
  pumi_disc->setCoordinates(x);
}

void updateRefConfig (
  const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& pumi_disc,
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
  const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& pumi_disc,
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
  const Teuchos::RCP<AlbPUMI::AbstractPUMIDiscretization>& pumi_disc,
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

    ++it;
    const long n_negative_simplices = apf::verifyVolumes(
      pumi_disc->getFMDBMeshStruct()->getMesh());
    // Adjust alpha bounds.
    if (n_negative_simplices == 0)
      alpha_lo = alpha;
    else
      alpha_hi = alpha;

    if (Teuchos::DefaultComm<int>::getComm()->getRank() == 0)
      std::cout << "amb: findAlpha iteration " << it
                << " n_negative_simplices " << n_negative_simplices
                << " alpha " << alpha
                << " in [" << alpha_lo << ", " << alpha_hi << "]\n";

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
