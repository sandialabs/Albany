#ifndef ATO_PENALTYMODEL_HPP
#define ATO_PENALTYMODEL_HPP

#include "ATO_TopoTools.hpp"

namespace ATO {


    

/******************************************************************************/
template< typename N >
class PenaltyModel {
  public:
    PenaltyModel( Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl );
    virtual void Evaluate(Teuchos::Array<N>& topoVals, Teuchos::RCP<TopologyArray>& topologies,
                          int cell, int qp, N& response, Teuchos::Array<N>& dResponse)=0;
    virtual void getDependentFields(Teuchos::Array<PHX::MDField<N> >& depFields)=0;
    virtual void getDependentFields(Teuchos::Array< PHX::MDField<N>* >& depFields)=0;
    void getFieldDimensions(std::vector<int>& dims);
  protected:
    int numDims, rank;
    PHX::MDField<N> gradX;
};
/******************************************************************************/


/******************************************************************************/
template< typename N >
class PenaltyMixture : public PenaltyModel<N> {
  using PenaltyModel<N>::numDims;
  using PenaltyModel<N>::rank;
  using PenaltyModel<N>::gradX;
  public:
    PenaltyMixture( Teuchos::ParameterList& blockParams,
                    Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);
    void Evaluate(Teuchos::Array<N>& topoVals, Teuchos::RCP<TopologyArray>& topologies,
                  int cell, int qp, N& response, Teuchos::Array<N>& dResponse);
    void getDependentFields(Teuchos::Array<PHX::MDField<N> >& depFields);
    void getDependentFields(Teuchos::Array< PHX::MDField<N>* >& depFields);
  private:
    int topologyIndex;
    int functionIndex;
    Teuchos::Array<int> materialIndices;
    Teuchos::Array<int> mixtureTopologyIndices;
    Teuchos::Array<int> mixtureFunctionIndices;
    Teuchos::Array<PHX::MDField<N> > workConj;
};
/******************************************************************************/



/******************************************************************************/
template< typename N >
class PenaltyMaterial : public PenaltyModel<N> {
  using PenaltyModel<N>::numDims;
  using PenaltyModel<N>::rank;
  using PenaltyModel<N>::gradX;
  public:
    PenaltyMaterial( Teuchos::ParameterList& blockParams,
                     Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl);
    void Evaluate(Teuchos::Array<N>& topoVals, Teuchos::RCP<TopologyArray>& topologies,
                  int cell, int qp, N& response, Teuchos::Array<N>& dResponse);
    void getDependentFields(Teuchos::Array<PHX::MDField<N> >& depFields);
    void getDependentFields(Teuchos::Array< PHX::MDField<N>* >& depFields);
  private:
    int topologyIndex;
    int functionIndex;
    PHX::MDField<N> workConj;
};
/******************************************************************************/



/******************************************************************************/
template< typename N >
class PenaltyModelFactory {
public:
  Teuchos::RCP<PenaltyModel<N> > create(Teuchos::ParameterList& problemParams,
                                        const Teuchos::RCP<Albany::Layouts>& dl,
                                        std::string elementBlockName);
};
/******************************************************************************/

}
#include "ATO_PenaltyModel_Def.hpp"
#endif
