//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_WORKSET_HPP
#define PHAL_WORKSET_HPP

#include <list>

#include "Phalanx_ConfigDefs.hpp" // for std::vector
#include "Albany_DataTypes.hpp"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_DistributedParameterLibrary_Epetra.hpp"
#include <Intrepid_FieldContainer.hpp>

#include "Stokhos_OrthogPolyExpansion.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Stokhos_EpetraMultiVectorOrthogPoly.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_TypeKeyMap.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Comm.hpp"
#include "Epetra_Import.h"

typedef Albany::DistributedParameterLibrary<Epetra_Vector, Epetra_MultiVector> DistParamLib;

namespace PHAL {

struct Workset {

  typedef AlbanyTraits::EvalTypes ET;

  Workset() :
    transientTerms(false), accelerationTerms(false), ignore_residual(false) {}

  unsigned int numCells;
  unsigned int wsIndex;

  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> > sg_expansion;

  // These are solution related.
  Teuchos::RCP<const Epetra_Vector> x;
  Teuchos::RCP<const Epetra_Vector> xdot;
  Teuchos::RCP<const Epetra_Vector> xdotdot;
  Teuchos::RCP<ParamVec> params;
  Teuchos::RCP<const Epetra_MultiVector> Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot;
  Teuchos::RCP<const Epetra_MultiVector> Vxdotdot;
  Teuchos::RCP<const Epetra_MultiVector> Vp;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > sg_x;

  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > sg_xdot;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > sg_xdotdot;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > mp_x;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > mp_xdot;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > mp_xdotdot;

  // These are residual related.
  Teuchos::RCP<Epetra_Vector> f;
  Teuchos::RCP<Epetra_CrsMatrix> Jac;
  Teuchos::RCP<Epetra_MultiVector> JV;
  Teuchos::RCP<Epetra_MultiVector> fp;
  Teuchos::RCP<Epetra_MultiVector> fpV;
  Teuchos::RCP<Epetra_MultiVector> Vp_bc;
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > sg_Jac;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_JV;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_fp;
  Teuchos::RCP< Stokhos::ProductEpetraVector > mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > mp_Jac;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_JV;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_fp;

  Teuchos::RCP<const Albany::NodeSetList> nodeSets;
  Teuchos::RCP<const Albany::NodeSetCoordList> nodeSetCoords;

  Teuchos::RCP<const Albany::SideSetList> sideSets;

  // jacobian and mass matrix coefficients for matrix fill
  double j_coeff;
  double m_coeff; //d(x_dot)/dx_{new}
  double n_coeff; //d(x_dotdot)/dx_{new}

  // Current Time as defined by Rythmos
  double current_time;
  double previous_time;

  // flag indicating whether to sum tangent derivatives, i.e.,
  // compute alpha*df/dxdot*Vxdot + beta*df/dx*Vx + omega*df/dxddotot*Vxdotdot + df/dp*Vp or
  // compute alpha*df/dxdot*Vxdot + beta*df/dx*Vx + omega*df/dxdotdot*Vxdotdot and df/dp*Vp separately
  int num_cols_x;
  int num_cols_p;
  int param_offset;

  std::vector<int> *coord_deriv_indices;

  // Distributed parameter derivatives
  Teuchos::RCP<DistParamLib> distParamLib;
  std::string dist_param_deriv_name;
  bool transpose_dist_param_deriv;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > > local_Vp;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > dist_param_index;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >  wsElNodeEqID;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> >  wsCoords;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >  wsSHeight;
  Teuchos::ArrayRCP<double>  wsSphereVolume;
  Teuchos::ArrayRCP<double>  wsTemperature;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >  wsBasalFriction;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >  wsThickness;
  Teuchos::ArrayRCP<double>  wsFlowFactor;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsSurfaceVelocity;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsVelocityRMS;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > > >  ws_coord_derivs;
  std::string EBName;
  Teuchos::RCP<Albany::AbstractDiscretization> disc;

  Albany::StateArray* stateArrayPtr;
  Teuchos::RCP<Albany::EigendataStruct> eigenDataPtr;
  Teuchos::RCP<Epetra_MultiVector> auxDataPtr;

  bool transientTerms;
  bool accelerationTerms;

  // Flag indicating whether to ignore residual calculations in the
  // Jacobian calculation.  This only works for some problems where the
  // the calculation of the Jacobian doesn't require calculation of the
  // residual (such as linear problems), but if it does work it can
  // significantly reduce Jacobian calculation cost.
  bool ignore_residual;

  // Flag indicated whether we are solving the adjoint operator or the
  // forward operator.  This is used in the Albany application when
  // either the Jacobian or the transpose of the Jacobian is scattered.
  bool is_adjoint;

  // New field manager response stuff
  Teuchos::RCP<const Teuchos::Comm<int> > comm;
  Teuchos::RCP<const Epetra_Import> x_importer;
  Teuchos::RCP<Epetra_Vector> g;
  Teuchos::RCP<Epetra_MultiVector> dgdx;
  Teuchos::RCP<Epetra_MultiVector> dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> dgdxdotdot;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdx;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdxdotdot;
  Teuchos::RCP<Epetra_MultiVector> dgdp;
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > sg_g;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_dgdx;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_dgdxdot;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_dgdxdotdot;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > overlapped_sg_dgdx;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > overlapped_sg_dgdxdot;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > overlapped_sg_dgdxdotdot;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_dgdp;
  Teuchos::RCP< Stokhos::ProductEpetraVector > mp_g;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_dgdx;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_dgdxdot;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_dgdxdotdot;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > overlapped_mp_dgdx;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > overlapped_mp_dgdxdot;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > overlapped_mp_dgdxdotdot;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_dgdp;

  // Meta-function class encoding T<EvalT::ScalarT> given EvalT
  // where T is any lambda expression (typically a placeholder expression)
  template <typename T>
  struct ApplyEvalT {
    template <typename EvalT> struct apply {
      typedef typename boost::mpl::apply<T, typename EvalT::ScalarT>::type type;
    };
  };

  // Meta-function class encoding RCP<ValueTypeSerializer<int,T> > for a given
  // type T.  This is to eliminate an error when using a placeholder expression
  // for the same thing in CreateLambdaKeyMap below
  struct ApplyVTS {
    template <typename T>
    struct apply {
      typedef Teuchos::RCP< Teuchos::ValueTypeSerializer<int,T> > type;
    };
  };

  // mpl::vector mapping evaluation type EvalT to serialization class
  // ValueTypeSerializer<int, EvalT::ScalarT>, which is used for MPI
  // communication of scalar types.
  typedef PHAL::CreateLambdaKeyMap<AlbanyTraits::BEvalTypes,
                                   ApplyEvalT<ApplyVTS> >::type SerializerMap;

  // Container storing serializers for each evaluation type
  PHAL::TypeKeyMap<SerializerMap> serializerManager;

  void print(std::ostream &os){

    os << "Printing workset data:" << std::endl;
    os << "\tEB name : " << EBName << std::endl;
    os << "\tnumCells : " << numCells << std::endl;
    os << "\twsElNodeEqID : " << std::endl;
    for(int i = 0; i < wsElNodeEqID.size(); i++)
      for(int j = 0; j < wsElNodeEqID[i].size(); j++)
        for(int k = 0; k < wsElNodeEqID[i][j].size(); k++)
          os << "\t\twsElNodeEqID[" << i << "][" << j << "][" << k << "] = " <<
            wsElNodeEqID[i][j][k] << std::endl;
    os << "\twsCoords : " << std::endl;
    for(int i = 0; i < wsCoords.size(); i++)
      for(int j = 0; j < wsCoords[i].size(); j++)
          os << "\t\tcoord0:" << wsCoords[i][j][0] << "][" << wsCoords[i][j][1] << std::endl;
  }

};

  template <typename EvalT> struct BuildSerializer {
    BuildSerializer(Workset& workset) {}
  };
  template <> struct BuildSerializer<PHAL::AlbanyTraits::Residual> {
    BuildSerializer(Workset& workset) {
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> > serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
      workset.serializerManager.
        setValue<PHAL::AlbanyTraits::Residual>(serializer);
    }
  };
  template <> struct BuildSerializer<PHAL::AlbanyTraits::Jacobian> {
    BuildSerializer(Workset& workset) {
      int num_nodes = workset.wsElNodeEqID[0].size();
      int num_eqns =  workset.wsElNodeEqID[0][0].size();
      int num_dof = num_nodes * num_eqns;
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> >
        real_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,FadType> > serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,FadType>(
                       real_serializer, num_dof));
      workset.serializerManager.
        setValue<PHAL::AlbanyTraits::Jacobian>(serializer);
    }
  };
  template <> struct BuildSerializer<PHAL::AlbanyTraits::Tangent> {
    BuildSerializer(Workset& workset) {
      int num_cols_tot = workset.param_offset + workset.num_cols_p;
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> >
        real_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,TanFadType> > serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,TanFadType>(
                       real_serializer, num_cols_tot));
      workset.serializerManager.
        setValue<PHAL::AlbanyTraits::Tangent>(serializer);
    }
  };
#ifdef ALBANY_SG_MP
  template <> struct BuildSerializer<PHAL::AlbanyTraits::SGResidual> {
    BuildSerializer(Workset& workset) {
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> >
        real_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,SGType> > serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,SGType>(
                       workset.sg_expansion, real_serializer));
      workset.serializerManager.
        setValue<PHAL::AlbanyTraits::SGResidual>(serializer);
    }
  };
  template <> struct BuildSerializer<PHAL::AlbanyTraits::SGJacobian> {
    BuildSerializer(Workset& workset) {
      int num_nodes = workset.wsElNodeEqID[0].size();
      int num_eqns =  workset.wsElNodeEqID[0][0].size();
      int num_dof = num_nodes * num_eqns;
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> >
        real_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,SGType> > sg_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,SGType>(
                       workset.sg_expansion, real_serializer));
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,SGFadType> > serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,SGFadType>(
                       sg_serializer, num_dof));
      workset.serializerManager.
        setValue<PHAL::AlbanyTraits::SGJacobian>(serializer);
    }
  };
  template <> struct BuildSerializer<PHAL::AlbanyTraits::SGTangent> {
    BuildSerializer(Workset& workset) {
      int num_cols_tot = workset.param_offset + workset.num_cols_p;
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> >
        real_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,SGType> > sg_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,SGType>(
                       workset.sg_expansion, real_serializer));
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,SGFadType> > serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,SGFadType>(
                       sg_serializer, num_cols_tot));
      workset.serializerManager.
        setValue<PHAL::AlbanyTraits::SGTangent>(serializer);
    }
  };
  template <> struct BuildSerializer<PHAL::AlbanyTraits::MPResidual> {
    BuildSerializer(Workset& workset) {
      int nblock = workset.mp_x->size();
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> >
        real_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,MPType> > serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,MPType>(
                       real_serializer, nblock));
      workset.serializerManager.
        setValue<PHAL::AlbanyTraits::MPResidual>(serializer);
    }
  };
  template <> struct BuildSerializer<PHAL::AlbanyTraits::MPJacobian> {
    BuildSerializer(Workset& workset) {
      int nblock = workset.mp_x->size();
      int num_nodes = workset.wsElNodeEqID[0].size();
      int num_eqns =  workset.wsElNodeEqID[0][0].size();
      int num_dof = num_nodes * num_eqns;
       Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> >
         real_serializer =
         Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
       Teuchos::RCP< Teuchos::ValueTypeSerializer<int,MPType> > mp_serializer =
         Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,MPType>(
                        real_serializer, nblock));
       Teuchos::RCP< Teuchos::ValueTypeSerializer<int,MPFadType> > serializer =
         Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,MPFadType>(
                        mp_serializer, num_dof));
       workset.serializerManager.
         setValue<PHAL::AlbanyTraits::MPJacobian>(serializer);
    }
  };
  template <> struct BuildSerializer<PHAL::AlbanyTraits::MPTangent> {
    BuildSerializer(Workset& workset) {
      int nblock = workset.mp_x->size();
      int num_cols_tot = workset.param_offset + workset.num_cols_p;
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> >
        real_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,MPType> > mp_serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,MPType>(
                       real_serializer, nblock));
      Teuchos::RCP< Teuchos::ValueTypeSerializer<int,MPFadType> > serializer =
        Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,MPFadType>(
                       mp_serializer, num_cols_tot));
      workset.serializerManager.
        setValue<PHAL::AlbanyTraits::MPTangent>(serializer);
    }
  };
#endif //ALBANY_SG_MP

}

#endif
