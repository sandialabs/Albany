// Copyright (c) 2013, Sandia Corporation.
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
// 
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
// 
//     * Neither the name of Sandia Corporation nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

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

