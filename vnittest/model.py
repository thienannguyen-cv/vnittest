# ======================================================
# Model: Data loading and processing of gradients
# ======================================================
import pickle
import numpy as np
from matplotlib import cm as cm
from matplotlib import colors as mcolors

class GradientModel:
    def __init__(self, debug_folder):
        from vnittest import utils
        
        self.debug_folder = debug_folder
        self.is_explicit_mode = None
        
        self.in_ch_id = 0
        self.in_z_id = 0
        self.in_row = 0
        self.in_col = 0
        self.in_val = 0
        self.out_ch_id = 0
        self.out_z_id = 0
        self.out_row = 0
        self.out_col = 0
        self.out_val = 0
        self.input_ch_id = 0
        self.input_z_id = 0
        self.input_row = 0
        self.input_col = 0
        self.input_val = 0
        self.target_ch_id = 0
        self.target_z_id = 0
        self.target_row = 0
        self.target_col = 0
        self.target_val = 0
        
    def allocate_resource(self):
        self._load_data()
        self.progress_bar.set_progress(5)
        self._process_gradients()
        self.progress_bar.set_progress(15)
        self._calc_flow_matrices()
        self.initialize_limits()
        self.progress_bar.set_progress(18)

    def _load_data(self):
        with open(self.debug_folder + 'flow_info.pkl', 'rb') as file:
            self.flow_data = pickle.load(file)
        self.layer_activation_gradients = self.flow_data["activation_gradients"]
        self.inter_layer_gradient_flows = (self.flow_data["gradient_flows"])
        self.input_representation = np.load(self.debug_folder + 'input_representation.npy')
        self.target_representation = np.load(self.debug_folder + 'target_representation.npy')
        # Copy for editing (e.g. interactive debug) so original data remains unchanged.
        self.editable_input_representation = self.input_representation.copy()
        
        self.is_explicit_mode = ((len(self.input_representation.shape)==3) and ((self.input_representation.shape[0])==(3*3)))
        
    def reload_model(self, input_fpath):
        with open(self.debug_folder + 'flow_info.pkl', 'rb') as file:
            self.flow_data = pickle.load(file)
        self.layer_activation_gradients = self.flow_data["activation_gradients"]
        self.inter_layer_gradient_flows = self.flow_data["gradient_flows"]
        self.input_representation = np.load(self.debug_folder + input_fpath)
        self.target_representation = np.load(self.debug_folder + 'target_representation.npy')
        
        self._process_gradients()
        self._calc_flow_matrices()
        
    def _process_gradients(self):
        self.flattened_gradients = {}  # mapping: layer -> flattened gradient vector
        self.all_layers = []
        self.selectable_layers = []
        # Process activation gradients
        for layer_key, grad_matrix in self.layer_activation_gradients.items():
            self.all_layers.append(layer_key)
            if not isinstance(layer_key, tuple):
                if self.is_explicit_mode:
                    grad_matrix = grad_matrix.sum(axis=0,keepdims=False)
                self.flattened_gradients[layer_key] = grad_matrix.reshape(-1)
        # Process gradient flows between layers
        for flow_key, flow_val in self.inter_layer_gradient_flows.items():
            self.all_layers.append(flow_key[0])
            self.all_layers.append(flow_key[1])
            if not isinstance(flow_val, dict):
                if flow_val is not None:
                    if flow_key[0] not in self.flattened_gradients:
                        self.flattened_gradients[flow_key[0]] = flow_val.sum(axis=-1, keepdims=False)
                        if self.is_explicit_mode:
                            self.flattened_gradients[flow_key[0]] = (self.flattened_gradients[flow_key[0]]).sum(axis=0, keepdims=False)
                    if flow_key[1] not in self.flattened_gradients:
                        self.flattened_gradients[flow_key[1]] = flow_val.sum(axis=-2, keepdims=False)
                        if self.is_explicit_mode:
                            self.flattened_gradients[flow_key[1]] = (self.flattened_gradients[flow_key[1]]).sum(axis=0, keepdims=False)
            else:
                H, W = ((self.layer_activation_gradients[flow_key[0]]).shape[-2]), ((self.layer_activation_gradients[flow_key[0]]).shape[-1])
                indices = list(flow_val["indices_in"])
                if not isinstance(indices[0], list):
                    if (indices[0]).shape[0] < (H*W):
                        indices0, indices1 = [], []
                        for idx in indices:
                            indices0.append(idx//W)
                            indices1.append(idx%W)
                        flow_val["indices_in"] = (indices0, indices1)
                H, W = ((self.layer_activation_gradients[flow_key[1]]).shape[-2]), ((self.layer_activation_gradients[flow_key[1]]).shape[-1])
                indices = list(flow_val["indices_out"])
                if not isinstance(indices[0], list):
                    if (indices[0]).shape[0] < (H*W):
                        indices0, indices1 = [], []
                        for idx in indices:
                            indices0.append(idx//W)
                            indices1.append(idx%W)
                        flow_val["indices_out"] = (indices0, indices1)
            self.selectable_layers.append(flow_key[0])
            
        self.all_layers = sorted(set(self.all_layers))
        self.selectable_layers = sorted(set(self.selectable_layers))
        
        # Build global node indices and node positions for each layer
        self.global_head_indices = {}
        self.global_node_indices = {}  # Mapping: (layer, (row, col)) -> unique global node index
        self.layer_node_positions = {} # Mapping for each layer: index -> (row, col)
        self.layer_color_maps = {}
        self.all_node_colors = []
        color_toggle = 0
        for layer_label in self.all_layers:
            gradient_vector = self.flattened_gradients[layer_label]
            if color_toggle % 2:
                color_list = [mcolors.to_hex(cm.viridis(i / len(gradient_vector))) for i in range(len(gradient_vector))]
            else:
                color_list = [mcolors.to_hex(cm.plasma(i / len(gradient_vector))) for i in range(len(gradient_vector))]
            self.layer_color_maps[layer_label] = color_list
            color_toggle += 1
            self.all_node_colors += color_list
        current_global_index = 0
        for i, layer_label in enumerate(self.all_layers):
            self.global_head_indices[layer_label] = current_global_index
            H, W = (self.layer_activation_gradients[layer_label].shape[-2]), (self.layer_activation_gradients[layer_label].shape[-1])
            self.layer_node_positions[layer_label] = [(r, c) for r in range(H) for c in range(W)]
            for j, (row, col) in enumerate(self.layer_node_positions[layer_label]):
                self.global_node_indices[(layer_label, (row, col))] = current_global_index
                current_global_index += 1

    def _calc_flow_matrices(self):
        flow_matrices = {}
        gradients = self.flattened_gradients
        if not self.is_explicit_mode:
            for source_layer, source_gradient in gradients.items():
                for target_layer, target_gradient in gradients.items():
                    layer_pair = (source_layer, target_layer)
                    if layer_pair in self.inter_layer_gradient_flows:
                        flow_value = self.inter_layer_gradient_flows[layer_pair]
                        if flow_value is None:
                            divisor = (0.*source_gradient.reshape(-1, 1)+1.) * target_gradient.reshape(1, -1)
                            flow_value = source_gradient.reshape(-1, 1) * target_gradient.reshape(1, -1)
                            flow_value[divisor!=0] = flow_value[divisor!=0] / divisor[divisor!=0] / divisor[divisor!=0]
                            divisor = (0.*flow_value+1.) * np.sum(np.abs(flow_value), axis=-1, keepdims=True)
                            flow_value = flow_value * np.sum(np.abs(flow_value), axis=-1, keepdims=True)
                            flow_value[divisor!=0] = flow_value[divisor!=0] / divisor[divisor!=0] / divisor[divisor!=0]
                        flow_matrices[layer_pair] = flow_value
        else:
            for source_layer, source_gradient in gradients.items():
                for target_layer, target_gradient in gradients.items():
                    layer_pair = (source_layer, target_layer)
                    if layer_pair in self.inter_layer_gradient_flows:
                        np.save(f"{source_layer}_conv.npy", self.layer_activation_gradients[source_layer])
                        np.save(f"{target_layer}_conv.npy", self.layer_activation_gradients[target_layer])
                        flow_matrices[layer_pair] = {}
            np.save(f"target_conv.npy", self.target_representation)
            np.save(f"input_conv.npy", self.input_representation)
        self.flow_matrices = flow_matrices

    def _calc_flow_threshold_limits(self):
        self.flow_threshold_min, self.flow_threshold_max = None, None
        if self.flow_matrices is not None:
            for key, matrix in self.flow_matrices.items():
                if matrix is not None:
                    if isinstance(matrix, dict):
                        if self.flow_threshold_min is None or 0 < self.flow_threshold_min:
                            self.flow_threshold_min = 0
                        if self.flow_threshold_max is None or 1 > self.flow_threshold_max:
                            self.flow_threshold_max = 1
                    else:
                        if self.flow_threshold_min is None or matrix.min() < self.flow_threshold_min:
                            self.flow_threshold_min = matrix.min()
                        if self.flow_threshold_max is None or matrix.max() > self.flow_threshold_max:
                            self.flow_threshold_max = matrix.max()
        if self.is_explicit_mode:
            self.flow_threshold_min = 1
            self.flow_threshold_max = 1
        if self.flow_threshold_min is None:
            self.flow_threshold_min = -1
        if self.flow_threshold_max is None:
            self.flow_threshold_max = -1
            
    def _calc_zoom_max(self):
        self.zoom_max = 0
        if self.is_explicit_mode:
            self.zoom_max = int(np.log2(np.sqrt(self.input_representation.shape[0])))*int(np.log2(self.input_representation.shape[-2]))
        
    def initialize_limits(self):
        self._calc_flow_threshold_limits()
        self._calc_zoom_max()
            
    def get_zoom_max(self):
        return self.zoom_max
            
    def set_progress_bar(self, progress_bar):
        self.progress_bar = progress_bar
