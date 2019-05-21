#!/bin/bash
 
if [ ! -d Result_04_ir ];then
   mkdir Result_04_ir
fi

mo_tf.py --input_model ./Result_03_frozen/frozen.pb   --output_dir ./Result_04_ir --data_type FP16

