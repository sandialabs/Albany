//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
//
// To do:
//  1.  Expand to symmetry group (See Nye).

#include <MiniTensor.h>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "LocalNonlinearSolver.hpp"

namespace LCM {

/******************************************************************************/
template <typename EvalT, typename Traits>
LinearPiezoModel<EvalT, Traits>::LinearPiezoModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      C11(p->get<RealType>("C11")),
      C33(p->get<RealType>("C33")),
      C12(p->get<RealType>("C12")),
      C23(p->get<RealType>("C23")),
      C44(p->get<RealType>("C44")),
      C66(p->get<RealType>("C66")),
      e31(p->get<RealType>("e31")),
      e33(p->get<RealType>("e33")),
      e15(p->get<RealType>("e15")),
      E11(p->get<RealType>("Eps11")),
      E33(p->get<RealType>("Eps33"))
/******************************************************************************/
{
  // PARSE MATERIAL BASIS
  //
  R.set_dimension(num_dims_);
  R.clear();
  if (p->isType<Teuchos::ParameterList>("Material Basis")) {
    const Teuchos::ParameterList& pBasis =
        p->get<Teuchos::ParameterList>("Material Basis");
    if (pBasis.isType<Teuchos::Array<double>>("X axis")) {
      Teuchos::Array<double> Xhat =
          pBasis.get<Teuchos::Array<double>>("X axis");
      R(0, 0) = Xhat[0];
      R(1, 0) = Xhat[1];
      R(2, 0) = Xhat[2];
    }
    if (pBasis.isType<Teuchos::Array<double>>("Y axis")) {
      Teuchos::Array<double> Yhat =
          pBasis.get<Teuchos::Array<double>>("Y axis");
      R(0, 1) = Yhat[0];
      R(1, 1) = Yhat[1];
      R(2, 1) = Yhat[2];
    }
    if (pBasis.isType<Teuchos::Array<double>>("Z axis")) {
      Teuchos::Array<double> Zhat =
          pBasis.get<Teuchos::Array<double>>("Z axis");
      R(0, 2) = Zhat[0];
      R(1, 2) = Zhat[1];
      R(2, 2) = Zhat[2];
    }
  } else {
    R(0, 0) = 1.0;
    R(1, 1) = 1.0;
    R(2, 2) = 1.0;
  }

  if (p->isType<bool>("Test")) {
    test = p->get<bool>("Test");
  } else
    test = false;

  initializeConstants();

  // DEFINE THE EVALUATED FIELDS
  //
  stressName = "Stress";
  this->eval_field_map_.insert(std::make_pair(stressName, dl->qp_tensor));

  edispName = "Electric Displacement";
  this->eval_field_map_.insert(std::make_pair(edispName, dl->qp_vector));

  // DEFINE THE DEPENDENT FIELDS
  //
  strainName = "Strain";
  this->dep_field_map_.insert(std::make_pair(strainName, dl->qp_tensor));

  efieldName = "Electric Potential Gradient";
  this->dep_field_map_.insert(std::make_pair(efieldName, dl->qp_vector));

  // DEFINE STATE VARIABLES (output)
  //

  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(stressName);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // strain
  this->num_state_variables_++;
  this->state_var_names_.push_back(strainName);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);

  // edisp
  this->num_state_variables_++;
  this->state_var_names_.push_back(edispName);
  this->state_var_layouts_.push_back(dl->qp_vector);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // efield
  this->num_state_variables_++;
  this->state_var_names_.push_back(efieldName);
  this->state_var_layouts_.push_back(dl->qp_vector);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
LinearPiezoModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
/******************************************************************************/
{
  auto strain = *dep_fields[strainName];
  auto Gradp  = *dep_fields[efieldName];

  auto stress = *eval_fields[stressName];
  auto edisp  = *eval_fields[edispName];

  int numCells = workset.numCells;

  if (num_dims_ == 1) {
  } else if (num_dims_ == 2) {
  } else if (num_dims_ == 3) {
    minitensor::Tensor<ScalarT> x(num_dims_), X(num_dims_);
    minitensor::Vector<ScalarT> E(num_dims_), D(num_dims_);
    // Compute Stress
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < num_pts_; ++qp) {
        if (test) {
          const ScalarT &x1 = strain(cell, qp, 0, 0),
                        &x2 = strain(cell, qp, 1, 1),
                        &x3 = strain(cell, qp, 2, 2),
                        &x4 = strain(cell, qp, 1, 2),
                        &x5 = strain(cell, qp, 0, 2),
                        &x6 = strain(cell, qp, 0, 1);
          const ScalarT &E1 = -Gradp(cell, qp, 0), &E2 = -Gradp(cell, qp, 1),
                        &E3 = -Gradp(cell, qp, 2);

          stress(cell, qp, 0, 0) = C11 * x1 + C12 * x2 + C23 * x3 - e31 * E3;
          stress(cell, qp, 1, 1) = C12 * x1 + C11 * x2 + C23 * x3 - e31 * E3;
          stress(cell, qp, 2, 2) = C23 * x1 + C23 * x2 + C33 * x3 - e33 * E3;
          stress(cell, qp, 1, 2) = C44 * x4 - e15 * E2;
          stress(cell, qp, 0, 2) = C44 * x5 - e15 * E1;
          stress(cell, qp, 0, 1) = C66 * x6;
          stress(cell, qp, 1, 0) = stress(cell, qp, 0, 1);
          stress(cell, qp, 2, 0) = stress(cell, qp, 0, 2);
          stress(cell, qp, 2, 1) = stress(cell, qp, 1, 2);

          edisp(cell, qp, 0) = e15 * x5 + E11 * E1;
          edisp(cell, qp, 1) = e15 * x4 + E11 * E2;
          edisp(cell, qp, 2) = e31 * x1 + e31 * x2 + e33 * x3 + E33 * E3;

        } else {
          x.fill(strain, cell, qp, 0, 0);
          E.fill(Gradp, cell, qp, 0);
          E *= -1.0;

          X = dotdot(C, x) - dot(E, e);
          D = dotdot(e, x) + dot(eps, E);

          for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) stress(cell, qp, i, j) = X(i, j);
          for (int i = 0; i < 3; i++) edisp(cell, qp, i) = D(i);
        }
      }
    }
  }
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
LinearPiezoModel<EvalT, Traits>::computeStateParallel(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::invalid_argument,
      ">>> ERROR (LinearPiezoModel): computeStateParallel not implemented");
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
LinearPiezoModel<EvalT, Traits>::initializeConstants()
/******************************************************************************/
{
  C.set_dimension(num_dims_);
  e.set_dimension(num_dims_);
  eps.set_dimension(num_dims_);

  minitensor::Tensor4<ScalarT> Ctmp(num_dims_);
  Ctmp.clear();
  minitensor::Tensor3<ScalarT> etmp(num_dims_);
  etmp.clear();
  minitensor::Tensor<ScalarT> epstmp(num_dims_);
  epstmp.clear();

  // create constants in tensor form
  Ctmp(0, 0, 0, 0) = C11;
  Ctmp(0, 0, 1, 1) = C12;
  Ctmp(0, 0, 2, 2) = C23;
  Ctmp(1, 1, 0, 0) = C12;
  Ctmp(1, 1, 1, 1) = C11;
  Ctmp(1, 1, 2, 2) = C23;
  Ctmp(2, 2, 0, 0) = C23;
  Ctmp(2, 2, 1, 1) = C23;
  Ctmp(2, 2, 2, 2) = C33;
  Ctmp(0, 1, 0, 1) = C66 / 2.0;
  Ctmp(1, 0, 1, 0) = C66 / 2.0;
  Ctmp(0, 2, 0, 2) = C44 / 2.0;
  Ctmp(2, 0, 2, 0) = C44 / 2.0;
  Ctmp(1, 2, 1, 2) = C44 / 2.0;
  Ctmp(2, 1, 2, 1) = C44 / 2.0;

  etmp(0, 0, 2) = e15 / 2.0;
  etmp(0, 2, 0) = e15 / 2.0;
  etmp(1, 1, 2) = e15 / 2.0;
  etmp(1, 2, 1) = e15 / 2.0;
  etmp(2, 0, 0) = e31;
  etmp(2, 1, 1) = e31;
  etmp(2, 2, 2) = e33;

  epstmp(0, 0) = E11;
  epstmp(1, 1) = E11;
  epstmp(2, 2) = E33;

  // rotate to requested basis
  C.clear();
  e.clear();
  eps.clear();
  for (int i = 0; i < num_dims_; i++)
    for (int j = 0; j < num_dims_; j++) {
      for (int k = 0; k < num_dims_; k++) {
        for (int l = 0; l < num_dims_; l++)
          for (int q = 0; q < num_dims_; q++)
            for (int r = 0; r < num_dims_; r++)
              for (int s = 0; s < num_dims_; s++)
                for (int t = 0; t < num_dims_; t++)
                  C(i, j, k, l) +=
                      Ctmp(q, r, s, t) * R(i, q) * R(j, r) * R(k, s) * R(l, t);
        for (int q = 0; q < num_dims_; q++)
          for (int r = 0; r < num_dims_; r++)
            for (int s = 0; s < num_dims_; s++)
              e(i, j, k) += etmp(q, r, s) * R(i, q) * R(j, r) * R(k, s);
      }
      for (int q = 0; q < num_dims_; q++)
        for (int r = 0; r < num_dims_; r++)
          eps(i, j) += epstmp(q, r) * R(i, q) * R(j, r);
    }
}

}  // namespace LCM
