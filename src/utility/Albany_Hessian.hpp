#ifndef ALBANY_HESSIAN_HPP
#define ALBANY_HESSIAN_HPP

#include "Albany_LinearOpWithSolveDecorators.hpp"
#include "Albany_DistributedParameter.hpp"
#include "Albany_ThyraTypes.hpp"

namespace Albany
{

  /**
   * \brief createDenseHessianLinearOp function
   *
   * This function computes the Thyra::LinearOp associated to
   * the Hessian w.r.t a parameter vector.
   *
   * \param p_vs [in] Thyra::VectorSpace which specifies the entries of the current parameter vector.
   */
  Teuchos::RCP<MatrixBased_LOWS> createDenseHessianLinearOp(
      Teuchos::RCP<const Thyra_VectorSpace> p_vs);

  /**
   * \brief createSparseHessianLinearOp function
   *
   * This function computes the Thyra::LinearOp associated to
   * the Hessian w.r.t a distributed parameter.
   *
   * \param p [in] Albany::DistributedParameter w.r.t which the Hessian is computed
   */
   Teuchos::RCP<Thyra_LinearOp> createSparseHessianLinearOp(
        const Teuchos::RCP<const DistributedParameter>& p);

  /**
   * \brief getHessianBlockIDs function
   *
   * This function gets the block IDs of a block of the Hessian matrix from a blockName.
   * For example, this function associates the following IDs to the following names:
   *  - i1=0, i2=0, and blockName="(x,x)",
   *  - i1=0, i2=1, and blockName="(x,p0)",
   *  - i1=0, i2=2, and blockName="(x,p1)",
   *  - i1=1, i2=0, and blockName="(p0,x)",
   *  - i1=3, i2=5, and blockName="(p2,p4)",
   *  - (...)
   *
   * \param i1 [out] First block index.
   *
   * \param i2 [out] Second block index.
   *
   * \param blockName [in] Name of the block for which the IDs have to be computed.
   */
  void getHessianBlockIDs(
      int &i1,
      int &i2,
      std::string blockName);

  /**
   * \brief getParameterVectorID function
   *
   * This function tests if the name parameterName is associated to a parameter vector
   * (in this case, a scalar parameter which is not included in a parameter vector in the .yaml file
   * is considered as a parameter vector of size 1) or to a distributed parameter.
   *
   * If the parameterName is associated to a parameter vector, the function computes the index of
   * associated parameter vector.
   *
   * \param i [out] Index associated to the parameter vector (if the current parameter is a parameter vector).
   *
   * \param is_distributed [out] Bool which specifies if the current parameter is a parameter vector.
   *
   * \param parameterName [in] Name of the current parameter.
   */
  void getParameterVectorID(
      int &i,
      bool &is_distributed,
      std::string parameterName);

} // namespace Albany

#endif // ALBANY_HESSIAN_HPP
