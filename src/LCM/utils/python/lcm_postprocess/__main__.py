import os
import sys
import lcm_postprocess

try:

    name_file_output_exodus = sys.argv[1]

except IndexError:

    raise IndexError('Name of input file required')

if os.path.exists(name_file_output_exodus) == False:

    raise IOError('File does not exist')


if len(sys.argv) == 3:

    lcm_postprocess.postprocess(name_file_output_exodus, plotting = sys.argv[2])

else:

    lcm_postprocess.postprocess(name_file_output_exodus, plotting = True)