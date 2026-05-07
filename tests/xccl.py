# Copyright 2016-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
# import intel_extension_for_pytorch as ipex

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29503'
    dist.init_process_group(backend='xccl', rank=rank, world_size=world_size)

def run_allreduce(rank, world_size):
    """Worker function that runs on each process."""
    setup(rank, world_size)

    # Device setup
    device = torch.device('xpu:{}'.format(rank))

    # Create a tensor filled with the rank's value, consistent size across ranks
    tensor = torch.ones(10).to(device) * rank

    # Perform the allreduce operation; sum of tensors across all ranks
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Define expected result: tensor of size 10 where every element is the sum of all ranks
    expected_tensor = torch.ones(10).to(device) * sum(range(world_size))

    # Print result of comparison of computed vs expected tensors
    is_successful = torch.allclose(tensor, expected_tensor)
    print(f"Rank {rank}: {'Success' if is_successful else 'Failed'} - {tensor}")

    # Clean up the distributed processes
    dist.destroy_process_group()

    # Return success status for pytest assertion
    return is_successful

def test_allreduce():
    """Pytest-compatible test function."""
    world_size = 2  # Number of processes
    mp.spawn(run_allreduce, args=(world_size,), nprocs=world_size, join=True)

# Add unittest wrapper for xmlrunner compatibility
class TestAllReduce(unittest.TestCase):
    def test_allreduce(self):
        """Unittest-compatible test method."""
        test_allreduce()

if __name__ == '__main__':
    # Support both unittest and direct execution
    import sys
    if 'unittest' in sys.modules or 'xmlrunner' in sys.modules:
        unittest.main()
    else:
        test_allreduce()
