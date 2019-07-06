#!/usr/bin/env python
"""
@author: peter.s
@project: MTANet
@time: 2019/6/25 21:44
@desc:
"""

from MTAN.model import MTANet

param_dict = {
    'cell_units':64,
    'enc_fwd_input_shapes':[[7, 6], [10, 6]],
    'enc_back_input_shapes':[[20, 6]],
    'dec_input_shape':[12, 1],
    'output_shape':[4, 1],
    'ext_feature_dim':0
}

model = MTANet(cell_units=64,
               enc_fwd_input_shapes=[[7, 6], [10, 6]],
               enc_back_input_shapes=[[20, 6]],
               dec_input_shape=[12, 1],
               output_shape=[4, 1],
               ext_feature_dim=0)

model.summary()
