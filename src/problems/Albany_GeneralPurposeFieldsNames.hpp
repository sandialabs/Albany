#ifndef ALBANY_GENERAL_PURPOSE_FIELDS_NAMES_HPP
#define ALBANY_GENERAL_PURPOSE_FIELDS_NAMES_HPP

#include <string>

namespace Albany
{

// Hard coding names for some fields that are used in many evaluators

static const std::string coord_vec_name        = "Coord Vec";
static const std::string weights_name          = "Weights";
static const std::string weighted_measure_name = "Weighted Measure";
static const std::string bf_name               = "BF";
static const std::string grad_bf_name          = "Grad BF";
static const std::string weighted_bf_name      = "wBF";
static const std::string weighted_grad_bf_name = "wGrad BF";
static const std::string jacobian_name         = "Jacobian";
static const std::string jacobian_det_name     = "Jacobian Det";
static const std::string jacobian_inv_name     = "Jacobian Inv";
static const std::string tangents_name         = "Tangents";
static const std::string metric_name           = "Metric";
static const std::string metric_det_name       = "Metric Det";
static const std::string metric_inv_name       = "Metric Inv";
static const std::string normal_name           = "Normal";

} // namespace Albany

#endif // ALBANY_GENERAL_PURPOSE_FIELDS_NAMES_HPP
