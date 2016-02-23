//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Albany_STKNodeSharing.hpp>
#include <stk_util/parallel/ParallelComm.hpp>
#include "Teuchos_TimeMonitor.hpp"

//----------------------------------------------------------------------

// AGS 03/2015: This is code from STK that was deprecated, so I moved it here
//              as part of Albany.

void Albany::fix_node_sharing(stk::mesh::BulkData& bulk_data) {

    TEUCHOS_FUNC_TIME_MONITOR("> Albany Setup: fix_node_sharing");

    stk::CommAll comm(bulk_data.parallel());

    for (int phase=0;phase<2;++phase)
    {
        for (int i=0;i<bulk_data.parallel_size();++i)
        {
            if ( i != bulk_data.parallel_rank() )
            {
                const stk::mesh::BucketVector& buckets = bulk_data.buckets(stk::topology::NODE_RANK);
                for (size_t j=0;j<buckets.size();++j)
                {
                    const stk::mesh::Bucket& bucket = *buckets[j];
                    if ( bucket.owned() )
                    {
                        for (size_t k=0;k<bucket.size();++k)
                        {
                            stk::mesh::EntityKey key = bulk_data.entity_key(bucket[k]);
                            comm.send_buffer(i).pack<stk::mesh::EntityKey>(key);
                        }
                    }
                }
            }
        }

        if (phase == 0 )
        {
            comm.allocate_buffers( bulk_data.parallel_size()/4 );
        }
        else
        {
            comm.communicate();
        }
    }

    for (int i=0;i<bulk_data.parallel_size();++i)
    {
        if ( i != bulk_data.parallel_rank() )
        {
            while(comm.recv_buffer(i).remaining())
            {
                stk::mesh::EntityKey key;
                comm.recv_buffer(i).unpack<stk::mesh::EntityKey>(key);
                stk::mesh::Entity node = bulk_data.get_entity(key);
                if ( bulk_data.is_valid(node) )
                {
                    bulk_data.add_node_sharing(node, i);
                }
            }
        }
    }
}

