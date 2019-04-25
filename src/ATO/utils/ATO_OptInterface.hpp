//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_OPT_INTERFACE_HPP
#define ATO_OPT_INTERFACE_HPP

#include <Teuchos_Array.hpp>
#include <string>

namespace ATO {

class OptInterface {
public:
  virtual void Compute(const double* p, double& g, double* dgdp, double& c, double* dcdp = nullptr) = 0;

  virtual void ComputeMeasure(const std::string& measureType, double& measure) = 0;

  virtual void ComputeMeasure(const std::string& measureType, const double* p,
                              double& measure, double* dmdp, const std::string& integrationMethod) = 0;

  void ComputeMeasure(const std::string& measureType, const double* p,
                      double& measure, const std::string& integrationMethod) {
    ComputeMeasure(measureType, p, measure, NULL, integrationMethod);
  }

  void ComputeMeasure(const std::string& measureType, const double* p, double& measure) {
    ComputeMeasure(measureType, p, measure, nullptr, "Gauss Quadrature");
  }

  void ComputeMeasure(const std::string& measureType, const double* p, double& measure, double* dmdp) {
    ComputeMeasure(measureType, p, measure, dmdp, "Gauss Quadrature");
  }

  virtual void InitializeOptDofs(double* p) = 0;
  virtual void getOptDofsLowerBound (Teuchos::Array<double>& b) const = 0;
  virtual void getOptDofsUpperBound (Teuchos::Array<double>& b) const = 0;

  virtual int GetNumOptDofs() const = 0;

  /* legacy */

  virtual void Compute(double* p, double& g, double* dgdp, double& c, double* dcdp = nullptr) = 0;
  virtual void ComputeConstraint(double* p, double& c, double* dcdp = nullptr) = 0;

  virtual void ComputeObjective(double* p, double& g, double* dgdp = nullptr) = 0;
  virtual void ComputeObjective(const double* p, double& g, double* dgdp = nullptr) = 0;
  virtual void ComputeVolume(double* p, const double* dfdp, double& v, double threshhold, double minP) = 0;

  /* end legacy */
};

} // namespace ATO

#endif // ATO_OPT_INTERFACE_HPP
