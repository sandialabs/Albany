//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_StringUtils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {


//**********************************************************************
template<typename EvalT, typename Traits>
AdvDiffResid<EvalT, Traits>::
AdvDiffResid(const Teuchos::ParameterList& p) :
  wBF     (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  wGradBF    (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Gradient Data Layout") ),
  U       (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  UGrad      (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  UDot       (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Residual   (p.get<std::string>                   ("Residual Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") )
{



  this->addDependentField(U.fieldTag());
  this->addDependentField(UGrad.fieldTag());
  this->addDependentField(UDot.fieldTag());
  this->addDependentField(wBF.fieldTag());
  this->addDependentField(wGradBF.fieldTag());

  this->addEvaluatedField(Residual);


  this->setName("AdvDiffResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];


  Teuchos::ParameterList* bf_list =
  p.get<Teuchos::ParameterList*>("Parameter List");


  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];


  beta = bf_list->get<Teuchos::Array<double>>("Advection Coefficient");
  TEUCHOS_TEST_FOR_EXCEPTION (static_cast<std::size_t>(beta.size())!=numDims, std::logic_error,
      "Error! The input array for 'Advection Coefficient' should have the same extent as the mesh num dims.\n"
      " - beta: [" << util::join(beta,",") << "]\n"
      " - num dims: " << numDims << "\n");

  mu = bf_list->get("Viscosity mu", 0.1);
  useAugForm = bf_list->get("Use Augmented Form", false);
  formType = bf_list->get("Augmented Form Type", 1);

  bool error = true;
  if (formType == 1 || formType == 2)
    error = false;

  TEUCHOS_TEST_FOR_EXCEPTION(error, std::logic_error,
       "Invalid Augmented Form Type: " << formType << "; valid options are 1 and 2");


  std::cout << "beta, mu: [" << util::join(beta,",") << "], " << ", " << mu << std::endl;
  std::cout << " vecDim = " << vecDim << std::endl;
  std::cout << " numDims = " << numDims << std::endl;
  std::cout << "Augmented Form? " << useAugForm << std::endl;
  if (useAugForm == true)
    std::cout << "  Augmented form type: " << formType << std::endl;


}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvDiffResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(UGrad,fm);
  this->utils.setFieldData(UDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AdvDiffResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  using ExecutionSpace = PHX::Device::execution_space;
  using Policy = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>;

  // Copy all the data needed in the kernel to local variables, so it can be captured by value
  TEUCHOS_TEST_FOR_EXCEPTION (numDims>3, std::logic_error,
      "Error! AdvDiffResid only supports up to 3 spatial dimensions.\n");
  double b[3] = {0.0, 0.0, 0.0};
  for (std::size_t idim=0; idim<numDims; ++idim)
    b[idim] = beta[idim];
  const double b0 = b[0], b1 = b[1], b2 = b[2];
  const double visc = mu;
  const int nqp   = numQPs;
  const int ndims = numDims;
  const int vdim  = vecDim;
  // 0: standard form; 1,2: augmented form type 1,2
  const int form = useAugForm ? formType : 0;

  const auto res      = Residual.get_static_view();
  const auto u        = U.get_static_view();
  const auto ugrad    = UGrad.get_static_view();
  const auto udot     = UDot.get_static_view();
  const auto w_bf     = wBF.get_static_view();
  const auto w_gradbf = wGradBF.get_static_view();

  const int numCells = workset.numCells;
  const int nnodes = numNodes;
  Kokkos::parallel_for(this->getName(), Policy({0,0},{numCells,nnodes}),
                       KOKKOS_LAMBDA(const int cell, const int node) {
    const double bt[3] = {b0, b1, b2};
    for (int i=0; i<vdim; i++)
      res(cell,node,i) = 0.0;

    if (form==0) { //standard form of advection-diffusion equation
      for (int qp=0; qp < nqp; ++qp) {
        //du/dt + beta*grad(u) - mu*delta(u) = 0
        res(cell,node,0) += udot(cell,qp,0)*w_bf(cell,node,qp);
        for (int idim=0; idim<ndims; ++idim) {
          res(cell,node,0) += bt[idim]*ugrad(cell,qp,0,idim)*w_bf(cell,node,qp) +
                              visc*ugrad(cell,qp,0,idim)*w_gradbf(cell,node,qp,idim);
        }
      }
    } else if (form==1) { //augmented form of advection diffusion equation, type 1
      for (int qp=0; qp < nqp; ++qp) {
        //du/dt + beta*q - mu*div(q) = 0
        res(cell,node,0) += udot(cell,qp,0)*w_bf(cell,node,qp);
        for (int idim=0; idim<ndims; ++idim) {
          res(cell,node,0) += bt[idim]*u(cell,qp,idim+1)*w_bf(cell,node,qp) +
                              visc*u(cell,qp,idim+1)*w_gradbf(cell,node,qp,idim);
        }
        //q - grad(u) = 0
        res(cell,node,1) += u(cell,qp,1)*w_bf(cell,node,qp) - ugrad(cell,qp,0,0)*w_bf(cell,node,qp);
        res(cell,node,2) += u(cell,qp,2)*w_bf(cell,node,qp) - ugrad(cell,qp,0,1)*w_bf(cell,node,qp);
      }
    } else { //augmented form of advection diffusion equation, type 2
      for (int qp=0; qp < nqp; ++qp) {
        //du/dt + q = 0
        res(cell,node,0) += udot(cell,qp,0)*w_bf(cell,node,qp) +
                            u(cell,qp,1)*w_bf(cell,node,qp) +
                            u(cell,qp,2)*w_bf(cell,node,qp);
        //q - beta*grad(u) + mu*delta(u) = 0
        for (int idim=0; idim<ndims; ++idim) {
          res(cell,node,idim+1) += u(cell,qp,idim+1)*w_bf(cell,node,qp) - bt[idim]*ugrad(cell,qp,0,idim)*w_bf(cell,node,qp)
                                 - visc*ugrad(cell,qp,0,idim)*w_gradbf(cell,node,qp,idim);
        }
      }
    }
  });
}

//**********************************************************************
}



