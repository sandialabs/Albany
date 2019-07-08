//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_SPATIAL_FILTER_HPP
#define ATO_SPATIAL_FILTER_HPP

#include "Albany_ThyraUtils.hpp"

#include "ATO_Types.hpp"

#include <set>

namespace Albany {
class Application;
class CombineAndScatterManager;
}

namespace ATO {

class OptimizationProblem;
class Topology;

// eventually make this a base class and derive from it to make
// various kernels.  Also add a factory.
class SpatialFilter {
public:
  using app_type = Albany::Application;
  using cas_type = Albany::CombineAndScatterManager;

  // Constructor(s)
  SpatialFilter(Teuchos::ParameterList& params);
  ~SpatialFilter () = default;

  void buildOperator (const app_type& app,
                      const cas_type& cas_manager);

  Teuchos::RCP<Thyra_LinearOp> getFilterOperator () const { return m_filterOperator; }

  int getNumIterations () const { return m_iterations; }
protected:

  using nbrs_map_type = std::map<GlobalPoint, std::set<GlobalPoint>>;

  void importNeighbors (nbrs_map_type&  neighbors,
                        const cas_type& cas_manager);

  int                           m_iterations;
  double                        m_filterRadius;

  Teuchos::RCP<Thyra_LinearOp>  m_filterOperator;
  Teuchos::Array<std::string>   m_blocks;
};

} // namespace ATO

#endif // ATO_SPATIAL_FILTER_HPP
