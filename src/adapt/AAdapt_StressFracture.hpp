//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if ! defined(AAdapt_StressFracture_hpp)
#define AAdapt_StressFracture_hpp

#include "Fracture.h"
#include "Albany_STKDiscretization.hpp"

namespace AAdapt {

///
/// \brief Stress based fracture criterion
///
/// This class contains the abstract interface for determining if
/// fracture has occurred between two elements, based on the average
/// stress in the elements sharing the edge between them.
///
class StressFracture: public LCM::AbstractFractureCriterion {

  public:

    ///
    /// \brief Default constructor for the criterion object
    ///
    StressFracture(int numDim_, stk_classic::mesh::EntityRank& elementRank,
                   const std::vector<std::vector<double> >& stresses,
                   double crit_stress, Albany::STKDiscretization& stk);


    ///
    /// \brief Stress fracture criterion function.
    ///
    /// \param[in] entity
    /// \param[in] probability
    /// \return is criterion met
    ///
    /// Given an entity and probability, will determine if fracture
    /// criterion is met. Will return true if fracture criterion is
    /// met, else false.  Fracture only defined on surface of
    /// elements. Thus, input entity must be of rank dimension-1, else
    /// error. For 2D, entity rank must = 1.  For 3D, entity rank must
    /// = 2.
    ///
    virtual
    bool
    computeFractureCriterion(stk_classic::mesh::Entity& entity, double p);

  private:

    StressFracture();
    StressFracture(const StressFracture&);
    StressFracture& operator=(const StressFracture&);

    const std::vector<std::vector<double> >& avg_stresses_;
    Albany::STKDiscretization& stk_;

    double critical_stress_;

}; // class StressFracture


} // namespace LCM

#endif // LCM_StressFracture_hpp
