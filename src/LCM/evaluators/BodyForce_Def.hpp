//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//
//
//
template <typename EvalT, typename Traits>
BodyForce<EvalT, Traits>::BodyForce(
    Teuchos::ParameterList&       p,
    Teuchos::RCP<Albany::Layouts> dl)
    : body_force_("Body Force", dl->qp_vector),
      density_(p.get<RealType>("Density")),
      weights_("Weights", dl->qp_scalar),
      coordinates_("Coord Vec", dl->qp_vector)
{
  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;

  std::vector<PHX::DataLayout::size_type> dims;

  vector_dl->dimensions(dims);

  num_qp_  = dims[1];
  num_dim_ = dims[2];

  std::string const& type = p.get("Body Force Type", "Constant");

  if (type == "Constant") {
    is_constant_    = true;
    constant_value_ = p.get<Teuchos::Array<RealType>>("Value");
  } else if (type == "Centripetal") {
    is_constant_     = false;
    rotation_center_ = p.get<Teuchos::Array<RealType>>(
        "Rotation Center", Teuchos::tuple<double>(0.0, 0.0, 0.0));
    rotation_axis_ = p.get<Teuchos::Array<RealType>>(
        "Rotation Axis", Teuchos::tuple<double>(0.0, 0.0, 0.0));
    angular_frequency_ = p.get<RealType>("Angular Frequency", 0.0);

    // Ensure that axisDirection is normalized
    double len = 0.0;
    for (int i = 0; i < 3; i++) {
      len += this->rotation_axis_[i] * this->rotation_axis_[i];
    }

    len = sqrt(len);
    for (int i = 0; i < 3; i++) { this->rotation_axis_[i] /= len; }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        "Invalid body force type " << type);
  }

  this->addDependentField(coordinates_);
  this->addDependentField(weights_);
  this->addEvaluatedField(body_force_);
  this->setName("Body Force" + PHX::typeAsString<EvalT>());
}

//
//
//
template <typename EvalT, typename Traits>
void
BodyForce<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(body_force_, fm);
  if (is_constant_ == false) this->utils.setFieldData(coordinates_, fm);
  if (is_constant_ == false) this->utils.setFieldData(weights_, fm);
}

//
//
//
template <typename EvalT, typename Traits>
void
BodyForce<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int const num_cells = workset.numCells;

  if (is_constant_ == true) {
    for (int cell = 0; cell < num_cells; ++cell) {
      for (int qp = 0; qp < num_qp_; ++qp) {
        for (int dim = 0; dim < num_dim_; ++dim) {
          body_force_(cell, qp, dim) = constant_value_[dim];
        }
      }
    }
  } else {
    double      omega2 = this->angular_frequency_ * this->angular_frequency_;
    ScalarT     qpmass, f_mag;
    MeshScalarT xyz[3], len2, dot, r, f_dir[3];

    for (int cell = 0; cell < num_cells; ++cell) {
      for (std::size_t qp = 0; qp < num_qp_; ++qp) {
        // Determine the qp's distance from the axis of rotation
        len2 = dot = 0.;
        for (std::size_t dim = 0; dim < num_dim_; dim++) {
          xyz[dim] = f_dir[dim] =
              this->coordinates_(cell, qp, dim) - this->rotation_center_[dim];
          dot += xyz[dim] * this->rotation_axis_[dim];
          len2 += xyz[dim] * xyz[dim];
        }
        r = std::sqrt(len2 - dot * dot);

        // Determine the direction of force due to centripetal acceleration
        len2 = 0.;
        for (std::size_t dim = 0; dim < num_dim_; dim++) {
          f_dir[dim] -= this->rotation_axis_[dim] * dot;
          len2 += f_dir[dim] * f_dir[dim];
        }
        MeshScalarT len_reciprocal = 1. / sqrt(len2);
        for (std::size_t dim = 0; dim < num_dim_; dim++) {
          f_dir[dim] *= len_reciprocal;
        }

        // Determine the qp's mass
        // qpmass = weights_(cell,qp) * density_(cell, qp);
        // qp volume * density - Is this right?
        qpmass = weights_(cell, qp) * density_;
        f_mag  = qpmass * omega2 * r;
        for (std::size_t dim = 0; dim < num_dim_; dim++)

          this->body_force_(cell, qp, dim) = f_dir[dim] * f_mag;
      }
    }
  }
}

}  // namespace LCM
