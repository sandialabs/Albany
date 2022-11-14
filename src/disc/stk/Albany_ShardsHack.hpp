#ifndef ALBANY_SHARDS_HACK_HPP
#define ALBANY_SHARDS_HACK_HPP

#include "Shards_BasicTopologies.hpp"

  extern "C" {
  typedef struct CellTopologyData_Subcell      Subcell ;
  }


namespace shards {

  template< class Traits > struct Descriptor ;

  template<>
  struct Descriptor<
    CellTopologyTraits< 0 , 1 , 1 ,
                        TypeListEnd , TypeListEnd ,
                        TypeListEnd , TypeListEnd ,
                        TypeListEnd > >
  {
    typedef CellTopologyTraits< 0 , 1 , 1 ,
                                TypeListEnd , TypeListEnd ,
                                TypeListEnd , TypeListEnd ,
                                TypeListEnd > Traits ;

    Subcell self ;

    CellTopologyData top ;

    Descriptor( const CellTopologyData * base , const char * name )
      {
        self.topology = & top ;
        self.node     = index_identity_array();

        top.base              = base ? base : & top ;
        top.name              = name ;
        top.key               = Traits::key ;
        top.dimension         = 0 ;
        top.vertex_count      = 1 ;
        top.node_count        = 1 ;
        top.edge_count        = 0 ;
        top.side_count        = 0 ;
        top.permutation_count = 0 ;
        top.subcell_homogeneity[0] = true ;
        top.subcell_homogeneity[1] = false ;
        top.subcell_homogeneity[2] = false ;
        top.subcell_homogeneity[3] = false ;
        top.subcell_count[0]       = 1 ;
        top.subcell_count[1]       = 0 ;
        top.subcell_count[2]       = 0 ;
        top.subcell_count[3]       = 0 ;
        top.subcell[0]             = & self ;
        top.subcell[1]             = NULL ;
        top.subcell[2]             = NULL ;
        top.subcell[3]             = NULL ;
        top.side                   = NULL ;
        top.edge                   = NULL ;
        top.permutation            = NULL ;
        top.permutation_inverse    = NULL ;
      };
  };

struct Node0D : public CellTopologyTraits<0,1,1>
{
#ifndef DOXYGEN_COMPILE
  typedef Node base ;
#endif /* DOXYGEN_COMPILE */
};

template<>
inline const CellTopologyData * getCellTopologyData<Node0D>()
{
  static const char name[] = "Node" ;
  static const Descriptor< Node0D::Traits > self( NULL , name );
  return & self.top ;
}

} // namespace shards

#endif // ALBANY_SHARDS_HACK_HPP
