//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef AERAS_STK_FROM_SPECTRAL_MESHSTRUCT_HPP
#define AERAS_STK_FROM_SPECTRAL_MESHSTRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_AbstractDiscretization.hpp"

//#include <Ionit_Initializer.h>

namespace Aeras {

  class SpectralOutputSTKMeshStruct : public Albany::GenericSTKMeshStruct {

    public:

//Constructor
    SpectralOutputSTKMeshStruct(
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const int numDim_, const int worksetSize_,
                  const bool periodic_, const double scale_,
                  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type& wsElNodeID_,
                  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& coords_,
                  const int points_per_edge_, const std::string element_name_);



    ~SpectralOutputSTKMeshStruct();

    void setFieldAndBulkData(
                  const Teuchos::RCP<const Teuchos_Comm>& commT,
                  const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const unsigned int neq_,
                  const Albany::AbstractFieldContainer::FieldContainerRequirements& req,
                  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                  const unsigned int worksetSize,
                  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& /*side_set_sis*/ = {},
                  const std::map<std::string,Albany::AbstractFieldContainer::FieldContainerRequirements>& /*side_set_req*/ = {});

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return false; }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return -1.0; }

    //Is this necessary here?
//    bool getInterleavedOrdering() const {return this->interleavedOrdering;}

    private:
    //Ioss::Init::Initializer ioInit;

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParametersQuads() const;

    Teuchos::RCP<const Teuchos::ParameterList>
      getValidDiscretizationParametersLines() const;

    Teuchos::RCP<Teuchos::FancyOStream> out;
    bool periodic;
    double scale;
    bool contigIDs; //boolean specifying if node / element / face IDs are contiguous; only relevant for 1 processor run
    const int numDim;
    const int points_per_edge;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type wsElNodeID;
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type coords;

    //Create enum type for the different kinds of elements (currently lines and quads)
    enum elemType {LINE, QUAD};
    elemType ElemType;

    protected:
  };

}
#endif
