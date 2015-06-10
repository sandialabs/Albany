//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: Epetra has been ifdef'ed out if ALBANY_EPETRA_EXE is off. 

#ifndef PHAL_WORKSET_HPP
#define PHAL_WORKSET_HPP

#include <list>

#include "Phalanx_config.hpp" // for std::vector
#include "Albany_DataTypes.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Import.h"
#endif
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_DistributedParameterLibrary_Tpetra.hpp"
#include <Intrepid_FieldContainer.hpp>

#include "Stokhos_OrthogPolyExpansion.hpp"
#if defined(ALBANY_EPETRA)
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Stokhos_EpetraMultiVectorOrthogPoly.hpp"
#endif

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_TypeKeyMap.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Comm.hpp"

typedef Albany::DistributedParameterLibrary<Tpetra_Vector, Tpetra_MultiVector, Albany::IDArray> DistParamLib;
typedef Albany::DistributedParameter<Tpetra_Vector, Tpetra_MultiVector, Albany::IDArray> DistParam;

#if defined(ALBANY_LCM)
// Forward declaration needed for Schwarz coupling
namespace Albany {
class Application;
} // namespace Albany
#endif

namespace PHAL {

struct Workset {

  typedef AlbanyTraits::EvalTypes ET;

  Workset() :
    transientTerms(false), accelerationTerms(false), ignore_residual(false) {}

  unsigned int numCells;
  unsigned int wsIndex;
  unsigned int numEqs;

  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> > sg_expansion;

#if defined(ALBANY_EPETRA)
  // These are solution related.
  Teuchos::RCP<const Epetra_Vector> x;
  Teuchos::RCP<const Epetra_Vector> xdot;
  Teuchos::RCP<const Epetra_Vector> xdotdot;
#endif
  //Tpetra analogs of x and xdot
  Teuchos::RCP<const Tpetra_Vector> xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT;
  
  Teuchos::RCP<ParamVec> params;
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<const Epetra_MultiVector> Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot;
  Teuchos::RCP<const Epetra_MultiVector> Vxdotdot;
  Teuchos::RCP<const Epetra_MultiVector> Vp;
#endif
  //Tpetra analogs of Vx, Vxdot, Vxdotdot and Vp
  Teuchos::RCP<const Tpetra_MultiVector> VxT;
  Teuchos::RCP<const Tpetra_MultiVector> VxdotT;
  Teuchos::RCP<const Tpetra_MultiVector> VxdotdotT;
  Teuchos::RCP<const Tpetra_MultiVector> VpT;
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > sg_x;

  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > sg_xdot;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > sg_xdotdot;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > mp_x;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > mp_xdot;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > mp_xdotdot;
#endif

#if defined(ALBANY_EPETRA)
  // These are residual related.
  Teuchos::RCP<Epetra_Vector> f;
#endif
  //Tpetra analog of f
  Teuchos::RCP<Tpetra_Vector> fT;
 
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_CrsMatrix> Jac;
#endif
  //Tpetra analog of Jac
  Teuchos::RCP<Tpetra_CrsMatrix> JacT;

#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_MultiVector> JV;
  Teuchos::RCP<Epetra_MultiVector> fp;
#endif
  //Tpetra analogs of JV and fp
  Teuchos::RCP<Tpetra_MultiVector> JVT;
  Teuchos::RCP<Tpetra_MultiVector> fpT;

#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_MultiVector> fpV;
  Teuchos::RCP<Epetra_MultiVector> Vp_bc;
#endif
  //Tpetra analogs of fpV and Vp_bc
  Teuchos::RCP<Tpetra_MultiVector> fpVT;
  Teuchos::RCP<Tpetra_MultiVector> Vp_bcT;

#if defined(ALBANY_EPETRA)
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > sg_Jac;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_JV;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_fp;
  Teuchos::RCP< Stokhos::ProductEpetraVector > mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > mp_Jac;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_JV;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_fp;
#endif

  Teuchos::RCP<const Albany::NodeSetList> nodeSets;
  Teuchos::RCP<const Albany::NodeSetCoordList> nodeSetCoords;

  Teuchos::RCP<const Albany::SideSetList> sideSets;

  // jacobian and mass matrix coefficients for matrix fill
  double j_coeff;
  double m_coeff; //d(x_dot)/dx_{new}
  double n_coeff; //d(x_dotdot)/dx_{new}

  // Current Time as defined by Rythmos
  double current_time;
  //amb Nowhere set. We should either set it or remove it.
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

  Kokkos::View<int***, PHX::Device> wsElNodeEqID_kokkos;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > >  wsElNodeEqID;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >  wsElNodeID;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> >  wsCoords;
  Teuchos::ArrayRCP<double>  wsSphereVolume;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > > >  ws_coord_derivs;
  std::string EBName;

  // Needed for Schwarz coupling and for dirichlet conditions based on dist parameters. 
  Teuchos::RCP<Albany::AbstractDiscretization> disc;
#if defined(ALBANY_LCM)
  // Needed for Schwarz coupling
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::Application> >
  apps_;

  Teuchos::RCP<Albany::Application>
  current_app_;
#endif

  Albany::StateArray* stateArrayPtr;
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Albany::EigendataStruct> eigenDataPtr;
  Teuchos::RCP<Epetra_MultiVector> auxDataPtr;
#endif

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
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<const Epetra_Import> x_importer;
#endif
  Teuchos::RCP<const Tpetra_Import> x_importerT;
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_Vector> g;
#endif
  //Tpetra analog of g
  Teuchos::RCP<Tpetra_Vector> gT;
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_MultiVector> dgdx;
  Teuchos::RCP<Epetra_MultiVector> dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> dgdxdotdot;
#endif
  //Tpetra analogs of dgdx and dgdxdot 
  Teuchos::RCP<Tpetra_MultiVector> dgdxT;
  Teuchos::RCP<Tpetra_MultiVector> dgdxdotT;
  Teuchos::RCP<Tpetra_MultiVector> dgdxdotdotT;
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdx;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdxdot;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdxdotdot;
#endif
  //Tpetra analogs of overlapped_dgdx and overlapped_dgdxdot
  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxT;
  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxdotT;
  Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdxdotdotT;
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Epetra_MultiVector> dgdp;
  Teuchos::RCP<Epetra_MultiVector> overlapped_dgdp;
#endif
  //Tpetra analog of dgdp
  Teuchos::RCP<Tpetra_MultiVector> dgdpT;
  //dp-convert Teuchos::RCP<Tpetra_MultiVector> overlapped_dgdpT;
#ifdef ALBANY_SG
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > sg_g;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_dgdx;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_dgdxdot;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_dgdxdotdot;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > overlapped_sg_dgdx;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > overlapped_sg_dgdxdot;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > overlapped_sg_dgdxdotdot;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > sg_dgdp;
#endif 
#ifdef ALBANY_ENSEMBLE 
  Teuchos::RCP< Stokhos::ProductEpetraVector > mp_g;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_dgdx;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_dgdxdot;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_dgdxdotdot;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > overlapped_mp_dgdx;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > overlapped_mp_dgdxdot;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > overlapped_mp_dgdxdotdot;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > mp_dgdp;
#endif

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

  template <> struct BuildSerializer<PHAL::AlbanyTraits::DistParamDeriv> {
     BuildSerializer(Workset& workset) {
       const Albany::IDArray& wsElNode =
           workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[0];
       int num_dof = wsElNode.dimension(1)*wsElNode.dimension(2);
       Teuchos::RCP< Teuchos::ValueTypeSerializer<int,RealType> >
         real_serializer =
         Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,RealType>);
       Teuchos::RCP< Teuchos::ValueTypeSerializer<int,TanFadType> > serializer =
         Teuchos::rcp(new Teuchos::ValueTypeSerializer<int,TanFadType>(
                        real_serializer, num_dof));
       workset.serializerManager.
         setValue<PHAL::AlbanyTraits::DistParamDeriv>(serializer);
     }
  };

#ifdef ALBANY_SG
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
#endif 
#ifdef ALBANY_ENSEMBLE 
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
#endif

}

#endif
