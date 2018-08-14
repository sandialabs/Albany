//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <MiniTensor.h>
#include <Phalanx_DataLayout.hpp>
#include <Teuchos_TestForException.hpp>
#include <cmath>
#include "Albany_MaterialDatabase.hpp"

namespace LCM {

template <typename T>
T
lcm_cbrt(T const& x)
{
  auto zero = Teuchos::ScalarTraits<T>::zero();

  if (x == zero) return zero;

  return std::cbrt(x);
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
HeliumODEs<EvalT, Traits>::HeliumODEs(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : total_concentration_(
          p.get<std::string>("Total Concentration Name"),
          dl->qp_scalar),
      delta_time_(p.get<std::string>("Delta Time Name"), dl->workset_scalar),
      diffusion_coefficient_(
          p.get<std::string>("Diffusion Coefficient Name"),
          dl->qp_scalar),
      he_concentration_(
          p.get<std::string>("He Concentration Name"),
          dl->qp_scalar),
      total_bubble_density_(
          p.get<std::string>("Total Bubble Density Name"),
          dl->qp_scalar),
      bubble_volume_fraction_(
          p.get<std::string>("Bubble Volume Fraction Name"),
          dl->qp_scalar)
{
  // get the material parameter lists
  // these are separate lists as defined in the Mechanics Problem
  // future work may consolidate into a single Material Parameters list
  Teuchos::ParameterList* mat_params_1 =
      p.get<Teuchos::ParameterList*>("Transport Parameters");
  Teuchos::ParameterList* mat_params_2 =
      p.get<Teuchos::ParameterList*>("Tritium Parameters");
  Teuchos::ParameterList* mat_params_3 =
      p.get<Teuchos::ParameterList*>("Molar Volume");

  avogadros_num_    = mat_params_1->get<RealType>("Avogadro's Number");
  t_decay_constant_ = mat_params_2->get<RealType>("Tritium Decay Constant");
  he_radius_        = mat_params_2->get<RealType>("Helium Radius");
  eta_              = mat_params_2->get<RealType>("Atoms Per Cluster");
  omega_            = mat_params_3->get<RealType>("Value");

  // add dependent fields
  this->addDependentField(total_concentration_);
  this->addDependentField(diffusion_coefficient_);
  this->addDependentField(delta_time_);

  // add evaluated fields
  this->addEvaluatedField(he_concentration_);
  this->addEvaluatedField(total_bubble_density_);
  this->addEvaluatedField(bubble_volume_fraction_);

  this->setName("Helium ODEs" + PHX::typeAsString<EvalT>());
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  num_pts_  = dims[1];
  num_dims_ = dims[2];

  total_concentration_name_ =
      p.get<std::string>("Total Concentration Name") + "_old";
  he_concentration_name_ = p.get<std::string>("He Concentration Name") + "_old";
  total_bubble_density_name_ =
      p.get<std::string>("Total Bubble Density Name") + "_old";
  bubble_volume_fraction_name_ =
      p.get<std::string>("Bubble Volume Fraction Name") + "_old";
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
HeliumODEs<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(total_concentration_, fm);
  this->utils.setFieldData(delta_time_, fm);
  this->utils.setFieldData(diffusion_coefficient_, fm);
  this->utils.setFieldData(he_concentration_, fm);
  this->utils.setFieldData(total_bubble_density_, fm);
  this->utils.setFieldData(bubble_volume_fraction_, fm);
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
HeliumODEs<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // Declaring time step & calculated input parameters
  ScalarT dt;
  ScalarT atomic_omega;
  // Declaring tangent, residual, norms, and increment for N-R
  minitensor::Tensor<ScalarT> tangent(3);
  minitensor::Vector<ScalarT> residual(3);
  ScalarT                     norm_residual_2, norm_residual_goal_2;
  minitensor::Vector<ScalarT> increment(3);

  // tolarences and iterations for newton
  const double tolerance = 1.0e-12, tolerance_2 = tolerance * tolerance;
  const int    maxIterations =
      20;  // FIXME: Currently a maximum, need relative measures
  // subincrementation for explicit predictor //FIXME: No guarantee of stability
  ScalarT   dt_explicit;
  const int explicit_sub_increments = 5;

  // state old
  Albany::MDArray total_concentration_old =
      (*workset.stateArrayPtr)[total_concentration_name_];
  Albany::MDArray he_concentration_old =
      (*workset.stateArrayPtr)[he_concentration_name_];
  Albany::MDArray total_bubble_density_old =
      (*workset.stateArrayPtr)[total_bubble_density_name_];
  Albany::MDArray bubble_volume_fraction_old =
      (*workset.stateArrayPtr)[bubble_volume_fraction_name_];

  // state new
  //   he_concentration_ - He concentration at t + deltat
  //   total_bubble_density - total bubble density at t + delta t
  //   bubble_volume_fraction - bubble volume fraction at t + delta t
  //   total_concentration_ - total concentration of tritium at t + delta t

  // fields required for computation
  //   diffusion_coefficient_ - current diffusivity (varies with temperature)
  //   delta_time_ - time step

  // input properties
  //   avogadros_num_ - Avogadro's Number
  //   omega_ - molar volume
  //   t_decay_constant_ - radioactive decay constant for tritium
  //   he_radius_ - radius of He atom
  //   eta_ - atoms per cluster (not variable)

  // convert molar volume to atomic volume through avogadros_num_
  atomic_omega = omega_ / avogadros_num_;

  // time step
  dt = delta_time_(0);

  // loop over cells and points for implicit time integration

  // constants for computations
  const double pi       = acos(-1.0);
  const double onethrd  = 1.0 / 3.0;
  const double twothrd  = 2.0 / 3.0;
  const double cub_tfpi = std::cbrt(3.0 / 4.0 / pi);

  // temporary variables
  ScalarT n1_old, nb_old, sb_old, n1, nb, sb;
  ScalarT n1_exp, nb_exp, sb_exp;
  ScalarT d, g_old, g;

  const double pi2             = pi * pi;
  const double cube_root_pi2   = std::cbrt(pi2);
  const double cube_root_2     = std::cbrt(2.0);
  const double cube_root_6     = std::cbrt(6.0);
  const double cube_root_9     = std::cbrt(9.0);
  const double cube_root_pi2_9 = std::cbrt(pi2 / 9.0);

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t pt = 0; pt < num_pts_; ++pt) {
      // temporary variables
      n1_old = he_concentration_old(cell, pt);
      nb_old = total_bubble_density_old(cell, pt);
      sb_old = bubble_volume_fraction_old(cell, pt);
      n1     = n1_old;
      nb     = nb_old;
      sb     = sb_old;
      d      = diffusion_coefficient_(cell, pt);

      // determine if any tritium exists - note that concentration is in mol
      // (not atoms) if no tritium exists, no need to solve the ODEs
      if (total_concentration_(cell, pt) > tolerance) {
        // source terms for helium bubble generation
        g_old = avogadros_num_ * t_decay_constant_ *
                total_concentration_old(cell, pt);
        g = avogadros_num_ * t_decay_constant_ * total_concentration_(cell, pt);

        // check if old bubble density is small
        // if small, use an explict guess to avoid issues with 1/nb and 1/sb in
        // tangent

        if (nb_old < tolerance) {
          // explicit time integration for predictor
          // Note that two or more steps are required to obtain a finite nb if
          // the total_concentration_old is zero.
          dt_explicit = dt / explicit_sub_increments;
          n1_exp      = n1_old;
          nb_exp      = nb_old;
          sb_exp      = sb_old;

          const ScalarT nb_exp2           = nb_exp * nb_exp;
          const ScalarT cube_root_nb_exp2 = lcm_cbrt(nb_exp2);

          for (int sub_increment = 0; sub_increment < explicit_sub_increments;
               sub_increment++) {
            n1 = n1_exp +
                 dt_explicit *
                     (g_old - 32.0 * pi * he_radius_ * d * n1_exp * n1_exp -
                      4.0 * pi * d * n1_exp * cub_tfpi * lcm_cbrt(sb_exp) *
                          cube_root_nb_exp2);
            nb = nb_exp +
                 dt_explicit * (16.0 * pi * he_radius_ * d * n1_exp * n1_exp);
            sb = sb_exp + atomic_omega / eta_ * dt_explicit *
                              (32. * pi * he_radius_ * d * n1_exp * n1_exp +
                               4.0 * pi * d * n1_exp * cub_tfpi *
                                   lcm_cbrt(sb_exp) * cube_root_nb_exp2);
            n1_exp = n1;
            nb_exp = nb;
            sb_exp = sb;
          }
        }

        ScalarT nb2           = nb * nb;
        ScalarT cube_root_nb2 = lcm_cbrt(nb2);
        ScalarT cube_root_sb  = lcm_cbrt(sb);

        // calculate initial residual for a relative tolerance
        residual(0) =
            n1 - n1_old -
            dt * (g - 32.0 * pi * he_radius_ * d * n1 * n1 -
                  4.0 * pi * d * n1 * cub_tfpi * cube_root_sb * cube_root_nb2);
        residual(1) = nb - nb_old - dt * (16.0 * pi * he_radius_ * d * n1 * n1);
        residual(2) =
            sb - sb_old -
            atomic_omega / eta_ * dt *
                (32. * pi * he_radius_ * d * n1 * n1 +
                 4.0 * pi * d * n1 * cub_tfpi * cube_root_sb * cube_root_nb2);
        norm_residual_2      = minitensor::norm_square(residual);
        norm_residual_goal_2 = tolerance_2 * norm_residual_2;
        int iter(0);

        // N-R loop for implicit time integration
        while (norm_residual_2 > norm_residual_goal_2 && iter < maxIterations) {
          // Common factors w/cube_root
          ScalarT cube_root_nb  = lcm_cbrt(nb);
          nb2                   = nb * nb;
          cube_root_nb2         = lcm_cbrt(nb2);
          cube_root_sb          = lcm_cbrt(sb);
          ScalarT sb2           = sb * sb;
          ScalarT cube_root_sb2 = lcm_cbrt(sb2);

          // calculate tangent
          tangent(0, 0) = 1.0 + 2.0 * dt * d *
                                    (32.0 * n1 * pi * he_radius_ +
                                     cube_root_6 * cube_root_nb2 *
                                         cube_root_pi2 * cube_root_sb);
          tangent(0, 1) = 4.0 * cube_root_2 * dt * d * n1 * cube_root_pi2 *
                          cube_root_sb / cube_root_9 / cube_root_nb;
          tangent(0, 2) = 2.0 * cube_root_2 * dt * d * n1 * cube_root_nb2 *
                          cube_root_pi2_9 / cube_root_sb2;
          tangent(1, 0) = -32.0 * dt * d * n1 * pi * he_radius_;
          tangent(1, 1) = 1.0;
          tangent(1, 2) = 0.0;
          tangent(2, 0) =
              -2.0 * dt * d * atomic_omega *
              (32.0 * n1 * pi * he_radius_ +
               cube_root_6 * cube_root_nb2 * cube_root_pi2 * cube_root_sb) /
              eta_;
          tangent(2, 1) = -4.0 * cube_root_2 * dt * d * n1 * atomic_omega *
                          cube_root_pi2 * cube_root_sb / cube_root_9 / eta_ /
                          cube_root_nb;
          tangent(2, 2) = 1.0 - 2.0 * cube_root_2 * dt * d * n1 *
                                    cube_root_nb2 * atomic_omega *
                                    cube_root_pi2_9 / eta_ / cube_root_sb2;

          // find increment
          increment = -minitensor::inverse(tangent) * residual;

          // update quantities
          n1 = n1 + increment(0);
          nb = nb + increment(1);
          sb = sb + increment(2);

          nb2           = nb * nb;
          cube_root_nb2 = lcm_cbrt(nb2);
          cube_root_sb  = lcm_cbrt(sb);

          // find new residual and norm
          residual(0) = n1 - n1_old -
                        dt * (g - 32. * pi * he_radius_ * d * n1 * n1 -
                              4.0 * pi * d * n1 * cub_tfpi * cube_root_sb *
                                  cube_root_nb2);
          residual(1) =
              nb - nb_old - dt * (16.0 * pi * he_radius_ * d * n1 * n1);
          residual(2) =
              sb - sb_old -
              atomic_omega / eta_ * dt *
                  (32. * pi * he_radius_ * d * n1 * n1 +
                   4.0 * pi * d * n1 * cub_tfpi * cube_root_sb * cube_root_nb2);
          norm_residual_2 = minitensor::norm_square(residual);
          iter++;
        }
      }

      // Update global fields

      he_concentration_(cell, pt)       = n1;
      total_bubble_density_(cell, pt)   = nb;
      bubble_volume_fraction_(cell, pt) = sb;
    }
  }
}
//------------------------------------------------------------------------------
}  // namespace LCM
