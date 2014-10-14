//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_BUCKETARRAY_HPP
#define ALBANY_BUCKETARRAY_HPP

#include <stk_mesh/base/Bucket.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FindRestriction.hpp>
#include <stk_mesh/base/CoordinateSystems.hpp>
#include <stk_mesh/base/MetaData.hpp>

#include <Shards_Array.hpp>

namespace Albany {

struct EntityDimension : public shards::ArrayDimTag {

  const char * name() const
  { static const char n[] = "EntityDimension" ; return n ; }

  static const EntityDimension & tag() ///< Singleton
  { static const EntityDimension self ; return self ; }

private:
  EntityDimension() {}
  EntityDimension( const EntityDimension & );
  EntityDimension & operator = ( const EntityDimension & );
};

template< class FieldType > struct BucketArray {};

/** \brief  \ref stk::mesh::Field "Field" data \ref shards::Array "Array"
 *          for a given scalar field and bucket
 */
template< typename ScalarType >
struct BucketArray< stk::mesh::Field<ScalarType,void,void,void,void,void,void,void> >
  : public
shards::Array<ScalarType,shards::FortranOrder,EntityDimension,void,void,void,void,void,void>
{
private:
  typedef unsigned char * byte_p ;
  BucketArray();
  BucketArray( const BucketArray & );
  BucketArray & operator = ( const BucketArray & );

public:

  typedef stk::mesh::Field<ScalarType,void,void,void,void,void,void,void> field_type ;
  typedef
  shards::Array<ScalarType,shards::FortranOrder,EntityDimension,void,void,void,void,void,void>
  array_type ;

  BucketArray( const field_type & f , const stk::mesh::Bucket & k )
  {
    if (k.field_data_is_allocated(f)) {
      array_type::assign( (ScalarType*)( k.field_data_location(f) ) ,
                          k.size() );

    }
  }
};

template<typename Tag>
inline size_t
get_size(stk::mesh::Bucket const&)
{
  return Tag::Size;
}

template <>
inline size_t
get_size<void>(stk::mesh::Bucket const&)
{
  return 0;
}

template <>
inline size_t
get_size<stk::mesh::Cartesian>(stk::mesh::Bucket const& b)
{
  return b.mesh().mesh_meta_data().spatial_dimension();
}

//----------------------------------------------------------------------
/** \brief  \ref stk::mesh::Field "Field" data \ref shards::Array "Array"
 *          for a given array field and bucket
 */
template< typename ScalarType ,
          class Tag1 , class Tag2 , class Tag3 , class Tag4 ,
          class Tag5 , class Tag6 , class Tag7 >
struct BucketArray< stk::mesh::Field<ScalarType,Tag1,Tag2,Tag3,Tag4,Tag5,Tag6,Tag7> >
  : public shards::ArrayAppend<
  shards::Array<ScalarType,shards::FortranOrder,Tag1,Tag2,Tag3,Tag4,Tag5,Tag6,Tag7> ,
  EntityDimension >::type
{
private:
  typedef unsigned char * byte_p ;
  BucketArray();
  BucketArray( const BucketArray & );
  BucketArray & operator = ( const BucketArray & );
public:

  typedef stk::mesh::Field<ScalarType,Tag1,Tag2,Tag3,Tag4,Tag5,Tag6,Tag7> field_type ;

  typedef typename shards::ArrayAppend<
    shards::Array<ScalarType,shards::FortranOrder,Tag1,Tag2,Tag3,Tag4,Tag5,Tag6,Tag7> ,
    EntityDimension >::type array_type ;

  BucketArray( const field_type & f , const stk::mesh::Bucket & b )
  {
    if ( b.field_data_is_allocated(f) ) {
      int stride[3];
      if (f.field_array_rank() == 1) {
        stride[0] = stk::mesh::field_scalars_per_entity(f, b);
      }
      else if (f.field_array_rank() == 2) {
        int dim0 = stk::mesh::find_restriction(f, b.entity_rank(), b.supersets()).dimension();
        stride[0] = dim0;
        stride[1] = stk::mesh::field_scalars_per_entity(f, b);
      }
      else if (f.field_array_rank() == 3) {
        int dim0 = stk::mesh::find_restriction(f, b.entity_rank(), b.supersets()).dimension();
        stride[0] = dim0;
        stride[1] = get_size<Tag2>(b) * dim0;
        stride[2] = stk::mesh::field_scalars_per_entity(f, b);
      }
      else {
        assert(false);
      }

      array_type::assign_stride(
        (ScalarType*)( b.field_data_location(f) ),
        stride,
        (typename array_type::size_type) b.size() );

    }
  }
};

} //namespace Albany

#endif // ALBANY_BUCKETARRAY_HPP
