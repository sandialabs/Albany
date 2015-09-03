//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
//
//
template<typename Residual, typename T, Intrepid::Index N = Intrepid::DYNAMIC>
std::unique_ptr<MiniSolver<Residual, T, N>>
nonlinearMethodFactory(NonlinearMethod const method_type)
{
  std::unique_ptr<MiniSolver<Residual, T, N>>
  method = nullptr;

  switch (method_type) {

  default:
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << std::endl;
    std::cerr << "Unknown nonlinear method.";
    std::cerr << std::endl;
    exit(1);
    break;

  case NonlinearMethod::NEWTON:
    method = new NewtonMiniSolver<Residual, T, N>();
    break;

  case NonlinearMethod::TRUST_REGION:
    break;

  case NonlinearMethod::CONJUGATE_GRADIENT:
    break;

  }

  return method;
}

//
//
//
template<typename Residual, typename T, Intrepid::Index N>
void
NewtonMiniSolver<Residual, T, N>::solve(
    Residual & residual,
    Intrepid::Vector<T, N> & soln)
{
  using AD = typename Sacado::Fad::DFad<T>;

  Intrepid::Index const
  dimension = soln.get_dimension();

  Intrepid::Vector<T, N>
  resi = residual.compute(soln);

  Intrepid::Vector<AD, N>
  soln_ad(dimension), resi_ad(dimension);

  for (Intrepid::Index i{0}; i < dimension; ++i) {
    soln_ad(i) = AD(dimension, i, soln(i));
  }

  T const
  initial_norm = Intrepid::norm(resi);

  this->num_iter_ = 0;

  this->converged_ = initial_norm <= this->abs_tol_;

  while (this->converged_ == false) {

    resi_ad = residual.compute(soln_ad);

    resi = Sacado::Value<Intrepid::Vector<AD, N>>::eval(resi_ad);

    this->abs_error_ = Intrepid::norm(resi);

    this->rel_error_ = this->abs_error_ / initial_norm;

    bool const
    converged_relative = this->rel_error_ <= this->rel_tol_;

    bool const
    converged_absolute = this->abs_error_ <= this->abs_tol_;

    this->converged_ = converged_relative || converged_absolute;

    bool const
    is_max_iter = this->num_iter_ >= this->max_num_iter_;

    bool const
    end_solve = this->converged_ || is_max_iter;

    if (end_solve == true) break;

    Intrepid::Tensor<T, N>
    Hessian(dimension);

    for (Intrepid::Index i{0}; i < dimension; ++i) {
      for (Intrepid::Index j{0}; j < dimension; ++j) {
        Hessian(i, j) = resi_ad(i).dx(j);
      }
    }

    Intrepid::Vector<T, N> const
    soln_incr = - Intrepid::solve(Hessian, resi);

    soln_ad += soln_incr;

    ++this->num_iter_;
  }

  soln = Sacado::Value<Intrepid::Vector<AD, N>>::eval(soln_ad);

  return;

}

} // namespace LCM
