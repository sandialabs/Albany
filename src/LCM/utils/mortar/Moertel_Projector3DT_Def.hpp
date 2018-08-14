//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_InterfaceT.hpp"
#include "Moertel_NodeT.hpp"
#include "Moertel_ProjectorT.hpp"
#include "Moertel_SegmentT.hpp"
#include "Moertel_UtilsT.hpp"

/*----------------------------------------------------------------------*
 |                                                           mwgee 10/05|
 | 3D case:                                                             |
 | this method evaluates the function                                   |
 | Fi(eta,alpha) = xs+alpha*n-Ni*xi = 0                                 |
 | and its gradient                                                     |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectorT)::
    evaluate_FgradF_3D_NodalNormal(
        double* F,
        double  dF[][3],
        const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
        MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
        double* eta,
        double  alpha,
        double& gap)
{
  // check the type of function on the segment
  // Here, we need a blinear triangle shape function
  MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)::FunctionType type =
      seg.FunctionType(0);
  if (type != MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)::func_LinearTri &&
      type != MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)::func_BiLinearQuad) {
    std::stringstream oss;
    oss << "***ERR*** MoertelT::Projector::evaluate_FgradF_3D_NodalNormal:\n"
        << "***ERR*** function is of wrong type\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  // evaluate the first function set on segment at eta
  int    nmnode = seg.Nnode();
  double val[100];
  double deriv[200];
  seg.EvaluateFunction(0, eta, val, nmnode, deriv);

  // get nodal coords of nodes and interpolate them
  MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)** mnodes = seg.Nodes();
  if (!mnodes) {
    std::stringstream oss;
    oss << "***ERR*** MoertelT::Projector::evaluate_FgradF_3D_NodalNormal:\n"
        << "***ERR*** segment " << seg.Id() << " ptr to it's nodes is zero\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  // interpolate Ni(eta)*xi
  // interpolate Ni,eta1(eta)*xi
  // interpolate Ni,eta2(eta)*xi
  double Nx[3];
  Nx[0] = Nx[1] = Nx[2] = 0.0;
  double Nxeta1[3];
  Nxeta1[0] = Nxeta1[1] = Nxeta1[2] = 0.0;
  double Nxeta2[3];
  Nxeta2[0] = Nxeta2[1] = Nxeta2[2] = 0.0;
  for (int i = 0; i < nmnode; ++i) {
    const double* X = mnodes[i]->XCoords();
    Nx[0] += val[i] * X[0];
    Nx[1] += val[i] * X[1];
    Nx[2] += val[i] * X[2];
    Nxeta1[0] += deriv[2 * i] * X[0];
    Nxeta1[1] += deriv[2 * i] * X[1];
    Nxeta1[2] += deriv[2 * i] * X[2];
    Nxeta2[0] += deriv[2 * i + 1] * X[0];
    Nxeta2[1] += deriv[2 * i + 1] * X[1];
    Nxeta2[2] += deriv[2 * i + 1] * X[2];
  }

  const double* X = node.XCoords();
  const double* n = node.Normal();

  // eval the function
  for (int i = 0; i < 3; ++i) F[i] = X[i] + alpha * n[i] - Nx[i];

  // build its gradient
  for (int i = 0; i < 3; ++i) {
    dF[i][0] = -Nxeta1[i];
    dF[i][1] = -Nxeta2[i];
    dF[i][2] = n[i];
  }

  gap =
      ((Nx[0] - X[0]) * n[0] + (Nx[1] - X[1]) * n[1] + (Nx[2] - X[2]) * n[2]) /
      sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);  // ||gap|| cos theta

  return true;
}

/*----------------------------------------------------------------------*
 |                                                           mwgee 10/05|
 |                                                 modded by gah 7/2010 |
 | 3D case:                                                             |
 | this method evaluates the function                                   |
 | Fi(eta,alpha) = Ni*xi+alpha*Ni*ni - xm = 0                           |
 | and its gradient                                                     |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectorT)::
    evaluate_FgradF_3D_SegmentNormal(
        double* F,
        double  dF[][3],
        const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
        MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
        double* eta,
        double  alpha,
        double& gap)
{
  // check the type of function on the segment
  // Here, we need a bilinear triangle shape function
  MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)::FunctionType type =
      seg.FunctionType(0);
  if (type != MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)::func_LinearTri &&
      type != MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)::func_BiLinearQuad) {
    std::stringstream oss;
    oss << "***ERR*** MoertelT::Projector::evaluate_F_3D_NodalNormal:\n"
        << "***ERR*** function is of wrong type\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  // evaluate the first function set on segment at eta
  int    nsnode = seg.Nnode();
  double val[100];
  double deriv[200];
  seg.EvaluateFunction(0, eta, val, nsnode, deriv);

  // get nodal coords of nodes and interpolate them
  MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)** snodes = seg.Nodes();
  if (!snodes) {
    std::stringstream oss;
    oss << "***ERR*** MoertelT::Projector::evaluate_FgradF_3D_SegmentNormal:\n"
        << "***ERR*** segment " << seg.Id() << " ptr to it's nodes is zero\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  // interpolate Ni(eta)*xi
  // interpolate Ni,eta1(eta)*xi
  // interpolate Ni,eta2(eta)*xi
  // interpolate Ni(eta)*ni
  // interpolate Ni,eta1(eta)*ni
  // interpolate Ni,eta2(eta)*ni
  double Nx[3];
  Nx[0] = Nx[1] = Nx[2] = 0.0;
  double Nxeta1[3];
  Nxeta1[0] = Nxeta1[1] = Nxeta1[2] = 0.0;
  double Nxeta2[3];
  Nxeta2[0] = Nxeta2[1] = Nxeta2[2] = 0.0;
  double Nn[3];
  Nn[0] = Nn[1] = Nn[2] = 0.0;
  double Nneta1[3];
  Nneta1[0] = Nneta1[1] = Nneta1[2] = 0.0;
  double Nneta2[3];
  Nneta2[0] = Nneta2[1] = Nneta2[2] = 0.0;
  for (int i = 0; i < nsnode; ++i) {
    const double* X = snodes[i]->XCoords();
    Nx[0] += val[i] * X[0];
    Nx[1] += val[i] * X[1];
    Nx[2] += val[i] * X[2];
    Nxeta1[0] += deriv[2 * i] * X[0];
    Nxeta1[1] += deriv[2 * i] * X[1];
    Nxeta1[2] += deriv[2 * i] * X[2];
    Nxeta2[0] += deriv[2 * i + 1] * X[0];
    Nxeta2[1] += deriv[2 * i + 1] * X[1];
    Nxeta2[2] += deriv[2 * i + 1] * X[2];

    const double* n = snodes[i]->Normal();
    Nn[0] += val[i] * n[0];
    Nn[1] += val[i] * n[1];
    Nn[2] += val[i] * n[2];
    Nneta1[0] += deriv[2 * i] * n[0];
    Nneta1[1] += deriv[2 * i] * n[1];
    Nneta1[2] += deriv[2 * i] * n[2];
    Nneta2[0] += deriv[2 * i + 1] * n[0];
    Nneta2[1] += deriv[2 * i + 1] * n[1];
    Nneta2[2] += deriv[2 * i + 1] * n[2];
  }

  const double* X = node.XCoords();

  // eval the function
  for (int i = 0; i < 3; ++i) F[i] = Nx[i] + alpha * Nn[i] - X[i];

  // build its gradient
  for (int i = 0; i < 3; ++i) {
    dF[i][0] = Nxeta1[i] + alpha * Nneta1[i];
    dF[i][1] = Nxeta2[i] + alpha * Nneta2[i];
    dF[i][2] = Nn[i];
  }

  gap =
      ((Nx[0] - X[0]) * Nn[0] + (Nx[1] - X[1]) * Nn[1] +
       (Nx[2] - X[2]) * Nn[2]) /
      sqrt(Nn[0] * Nn[0] + Nn[1] * Nn[1] + Nn[2] * Nn[2]);  // ||gap|| cos theta

  return true;
}
