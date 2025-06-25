import numpy as np

DEBUG_FLAG = True
OFF_THRESH = 0.5

is_show_color_bar = True

INDEX2OFFSETS = {
    0: (-1, -1), 1: (-1, 0), 2: (-1, 1),
    3: ( 0, -1), 4: ( 0, 0), 5: ( 0, 1),
    6: ( 1, -1), 7: ( 1, 0), 8: ( 1, 1)
}

OFFSETS2INDEX = {
    (-1, -1): 0, (-1, 0): 1,  (-1, 1): 2, 
    ( 0, -1): 3, ( 0, 0): 4,  ( 0, 1): 5, 
    ( 1, -1): 6, ( 1, 0): 7,  ( 1, 1): 8
 }

def shift_tensor(tensor, ch_offset):
    h, w = (tensor.shape[-2]), (tensor.shape[-1])
    
    dx, dy = INDEX2OFFSETS[ch_offset]
            
    z = tensor[ch_offset:(ch_offset+1),:,:,:]
    
    shifted = np.zeros_like(z)
    pad_x = (max(dx, 0), max(-dx, 0))
    pad_y = (max(dy, 0), max(-dy, 0))
    padded = np.pad(z, ((0, 0), (0, 0), pad_x, pad_y), mode='constant')
    shifted = padded[:, :, pad_x[1]:pad_x[1]+h, pad_y[1]:pad_y[1]+w]
    return shifted

def calculate_conv_ids(mask_ids, current_node_index, grid_shape):
        offset_x, offset_y = divmod(current_node_index, grid_shape[-1])
        mask = (((mask_ids[0])==offset_x) & ((mask_ids[1])==offset_y))
        offset_z, row, col = np.where(mask)
        return (offset_z[0]), (row[0]), (col[0])