# ======================================================
# Controller: Manage UI and interaction
# ======================================================
from vnittest import GradientModel, Visualizer
import pickle
import numpy as np
from ipyevents import Event
import ipywidgets as widgets
from IPython.display import display


class UIController:
    def __init__(self, model: GradientModel, visualizer: Visualizer):
        self.input_fpath = "input_representation.npy"
        self.model = model
        self.visualizer = visualizer
        
    def allocate_resource(self):
        self.zoom = 0
        self.is_conv1_choosed = True
        
        self.selected_layer = self.model.selectable_layers[0]
        self.connected_layer = self.selected_layer
        for flow_key in self.model.inter_layer_gradient_flows.keys():
            if (flow_key[0]) == self.selected_layer:
                self.connected_layer = (flow_key[1])
        
        # Region settings dictionary (can be updated via widgets)
        self.region_settings = {
            self.selected_layer: [
                0,
                self.model.layer_activation_gradients[self.selected_layer].shape[-1] - 1,
                0,
                self.model.layer_activation_gradients[self.selected_layer].shape[-2] - 1
            ],
            self.connected_layer: [
                0,
                self.model.layer_activation_gradients[self.connected_layer].shape[-1] - 1,
                0,
                self.model.layer_activation_gradients[self.connected_layer].shape[-2] - 1
            ]
        }
        self.selected_node_index = 0
        self.id_z = 0
        self.recticify_offset = 0
        self.current_threshold = (self.model.flow_threshold_max - self.model.flow_threshold_min) / 2

        # Offset parameters for target and input representations
        self.offset_params = {
            self.selected_layer: [0, 0],
            self.connected_layer: [0, 0]
        }
        
        self._allocate_heatmap_conv_resource()
        self._allocate_sankey_resource(force=True)
        self.progress_bar.set_progress(25)
        
        self.conv1, self.conv2, self.target_heatmap, self.input_heatmap, self.full_target_heatmap, self.full_input_heatmap = None, None, None, None, None, None
        
        self.build_widgets()
        self.setup_callbacks()
        
    def _calc_z(self, gradient_values, current_zoom):
        h, w = (gradient_values.shape[-2]), (gradient_values.shape[-1])
        z = gradient_values.reshape(-1,1,1,h,w)
        mask_ids = np.where(np.ones_like(z.reshape(-1,h,w)[0,:,:])==1)
        mask_ids_0 = (mask_ids[0]).reshape(1,1,1,h,w)
        mask_ids_1 = (mask_ids[1]).reshape(1,1,1,h,w)
        C_zoom = 1
        for i in range(int(current_zoom)):
            z = np.transpose(z.reshape(-1,C_zoom,C_zoom,h//2,2,w//2,2),axes=(0,1,4,2,6,3,5)).reshape(-1,C_zoom*2,C_zoom*2,h//2,w//2)
            mask_ids_0 = np.transpose(mask_ids_0.reshape(-1,C_zoom,C_zoom,h//2,2,w//2,2),axes=(0,1,4,2,6,3,5)).reshape(1,C_zoom*2,C_zoom*2,h//2,w//2)
            mask_ids_1 = np.transpose(mask_ids_1.reshape(-1,C_zoom,C_zoom,h//2,2,w//2,2),axes=(0,1,4,2,6,3,5)).reshape(1,C_zoom*2,C_zoom*2,h//2,w//2)
            h, w = (h//2), (w//2)
            C_zoom = C_zoom*2
            
        mask_ids = mask_ids_0.reshape(C_zoom*C_zoom,h,w), mask_ids_1.reshape(C_zoom*C_zoom,h,w)
        return z.reshape(-1,C_zoom*C_zoom,h,w), mask_ids
    
    def _original_resize(self, values, current_zoom):
        C, h, w = (values.shape[0]), (values.shape[1]), (values.shape[2])
        C_zoom = int(np.sqrt(C))
        v = values.reshape(C_zoom, C_zoom, h, w)
        for i in range(int(current_zoom)):
            C_zoom = C_zoom//2
            v = v.reshape(C_zoom,2,C_zoom,2,h,w).transpose(0,2,4,1,5,3)
            h, w = h*2, w*2
            v = v.reshape(C_zoom,C_zoom,h,w)
        return v.reshape(h,w)
    
    def _calc_region(self, mask_ids, region_settings):
        h, w = ((mask_ids[0]).shape[-2]), ((mask_ids[0]).shape[-1])
        offsets = np.where(np.ones_like(mask_ids[0])==1)
        mask = (((mask_ids[-2])>=(region_settings[2])) & ((mask_ids[-2])<=(region_settings[3])) & ((mask_ids[-1])>=(region_settings[0])) & ((mask_ids[-1])<=(region_settings[1])))
        offsets = ((offsets[-2]).reshape(-1,h,w)[mask]), ((offsets[-1]).reshape(-1,h,w)[mask])
        return int((offsets[-1]).min()), int((offsets[-1]).max()), int((offsets[-2]).min()), int((offsets[-2]).max())
    
    def _allocate_heatmap_conv_resource(self):
        self.conv_zs = {}
        self.conv_mask_ids = {}
        self.conv_region_settings = {}
        
        for layer_key, grad_matrix in self.model.layer_activation_gradients.items():
            z, mask_ids = self._calc_z(grad_matrix, self.zoom)
            self.conv_zs[layer_key] = z
            self.conv_mask_ids[layer_key] = mask_ids
            self.conv_region_settings[layer_key] = self._calc_region(mask_ids, self.region_settings[layer_key])
        
        z, mask_ids = self._calc_z(self.model.target_representation, self.zoom)
        self.conv_target = z
        z, mask_ids = self._calc_z(self.model.editable_input_representation, self.zoom)
        self.conv_editable_input = z
        
    def _allocate_sankey_resource(self, force=False):
        from vnittest import utils
        
        # Build lists for Sankey diagram: sources, targets, and flow values
        if force==True:
            if self.model.is_explicit_mode:
                self.sankey_sources, self.sankey_targets, self.sankey_values = [], [], []
                gradients = self.model.flattened_gradients
                for source_layer, source_gradient in gradients.items():
                    for target_layer, target_gradient in gradients.items():
                        layer_pair = (source_layer, target_layer)
                        if layer_pair in self.model.inter_layer_gradient_flows:
                            source_positions = (self.model.layer_node_positions[source_layer])
                            target_positions = (self.model.layer_node_positions[target_layer])
                            temp_shape = ((self.conv_mask_ids[source_layer])[0]).shape
                            pre_new_ids = np.where(np.ones_like((self.conv_mask_ids[source_layer])[0]))
                            new_ids = self._original_resize((pre_new_ids[0]).reshape(*temp_shape), self.zoom), self._original_resize((pre_new_ids[1]).reshape(*temp_shape), self.zoom), self._original_resize((pre_new_ids[2]).reshape(*temp_shape), self.zoom)
                            for i, (pre_r1, pre_c1) in enumerate(source_positions):
                                l1, r1, c1 = ((new_ids[0])[pre_r1, pre_c1]), ((new_ids[1])[pre_r1, pre_c1]), ((new_ids[2])[pre_r1, pre_c1])
                                for j in range(3*3):
                                    offsets = (utils.INDEX2OFFSETS[j])
                                    l2, r2, c2 = l1, (r1+(offsets[0])), (c1+(offsets[1]))
                                    if (r2>=0)  and (r2<(temp_shape[-2])) and (c2>=0)  and (c2<(temp_shape[-1])):
                                        flow_value = 0
                                            
                                        if np.abs((self.model.layer_activation_gradients[source_layer])[utils.OFFSETS2INDEX[(offsets[0], offsets[1])], ((self.conv_mask_ids[source_layer])[0])[l1,r1,c1], ((self.conv_mask_ids[source_layer])[1])[l1,r1,c1]])>0:
                                            flow_value = 1
                                        (self.model.flow_matrices[layer_pair])[(r1, c1), (r2, c2)] = flow_value
                                        self.sankey_sources.append(self.model.global_node_indices[(source_layer, (r1, c1))])
                                        self.sankey_targets.append(self.model.global_node_indices[(target_layer, (r2, c2))])
                                        self.sankey_values.append(flow_value)
            else:
                self.sankey_sources, self.sankey_targets, self.sankey_values = [], [], []
                for layer_pair, flow_matrix in self.model.flow_matrices.items():
                    source_layer, target_layer = layer_pair
                    source_positions = self.model.layer_node_positions[source_layer]
                    target_positions = self.model.layer_node_positions[target_layer]
                    if flow_matrix is not None:
                        if isinstance(flow_matrix, dict):
                            indices_in = (flow_matrix["indices_in"])
                            indices_out = (flow_matrix["indices_out"])
                            id_in = 0
                            id_out = 0
                            for i, (r1, c1) in enumerate(source_positions):
                                for j, (r2, c2) in enumerate(target_positions):
                                    while id_in<len(indices_in) and ((indices_in[0])[id_in]) < r1 and ((indices_in[1])[id_in]) < c1:
                                        id_in += 1
                                    while id_out<len(indices_out) and ((indices_out[0])[id_out]) < r2 and ((indices_out[1])[id_out]) < c2:
                                        id_out += 1
                                    flow_value = 0
                                    if (id_in<len(indices_in) and ((indices_in[0])[id_in]) == r1 and ((indices_in[1])[id_in]) == c1) or (id_out<len(indices_out) and ((indices_out[0])[id_out]) == r2 and ((indices_out[1])[id_out]) == c2):
                                        flow_value = 1
                                    self.sankey_sources.append(self.model.global_node_indices[(source_layer, (r1, c1))])
                                    self.sankey_targets.append(self.model.global_node_indices[(target_layer, (r2, c2))])
                                    self.sankey_values.append(flow_value)
                        else:
                            for i, (r1, c1) in enumerate(source_positions):
                                for j, (r2, c2) in enumerate(target_positions):
                                    flow_value = flow_matrix[i, j]
                                    if flow_value > 0:
                                        self.sankey_sources.append(self.model.global_node_indices[(source_layer, (r1, c1))])
                                        self.sankey_targets.append(self.model.global_node_indices[(target_layer, (r2, c2))])
                                        self.sankey_values.append(flow_value)

    def build_widgets(self):
        # Dropdown for layer selection
        self.layer_dropdown = widgets.Dropdown(
            options=self.model.selectable_layers,
            value=self.selected_layer,
            description='Select Layer:'
        )
        # Slider for node index
        max_node = int(np.prod(self.model.layer_activation_gradients[self.layer_dropdown.value].shape[-2:])) - 1
        self.node_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=max_node,
            step=1,
            description='Node Index:'
        )
        # Slider for threshold
        self.threshold_slider = widgets.FloatSlider(
            value=self.current_threshold,
            min=self.model.flow_threshold_min,
            max=self.model.flow_threshold_max,
            step=(self.model.flow_threshold_max - self.model.flow_threshold_min) / 100,
            description='Threshold:'
        )
        
        self.z_value_widget = widgets.IntText(value=1., description='Z:', layout=widgets.Layout(width='100px'), style={'description_width': '15px'})
        
        self.rectify_slider = widgets.IntSlider(
            value=4,
            min=0,
            max=9,
            step=1,
            description='Zoom:'
        )
        
        self.zoom_slider = widgets.FloatSlider(
            value=0,
            min=0,
            max=self.model.get_zoom_max(),
            step=1.,
            description='Zoom:'
        )

        # Offset widgets
        self.target_row_offset_widget = widgets.IntText(value=0, description='Target Row Offset:', layout=widgets.Layout(width='200px'), style={'description_width': '120px'})
        self.target_col_offset_widget = widgets.IntText(value=0, description='Target Col Offset:', layout=widgets.Layout(width='200px'), style={'description_width': '120px'})
        self.input_row_offset_widget  = widgets.IntText(value=0, description='Input Row Offset:', layout=widgets.Layout(width='200px'), style={'description_width': '120px'})
        self.input_col_offset_widget  = widgets.IntText(value=0, description='Input Col Offset:', layout=widgets.Layout(width='200px'), style={'description_width': '120px'})
        # Region selection widgets for Conv1/Input (Select)
        self.select_xmin_widget = widgets.IntText(value=(self.region_settings[self.selected_layer])[0], description='Select Xmin:', layout=widgets.Layout(width='150px'))
        self.select_xmax_widget = widgets.IntText(value=(self.region_settings[self.selected_layer])[1], description='Select Xmax:', layout=widgets.Layout(width='150px'))
        self.select_ymin_widget = widgets.IntText(value=(self.region_settings[self.selected_layer])[2], description='Select Ymin:', layout=widgets.Layout(width='150px'))
        self.select_ymax_widget = widgets.IntText(value=(self.region_settings[self.selected_layer])[3], description='Select Ymax:', layout=widgets.Layout(width='150px'))
        self.select_region_box = widgets.HBox([self.select_xmin_widget, self.select_xmax_widget, self.select_ymin_widget, self.select_ymax_widget])
        # Region selection widgets for Target (Output)
        self.output_xmin_widget = widgets.IntText(value=(self.region_settings[self.connected_layer])[0], description='Output Xmin:', layout=widgets.Layout(width='150px'))
        self.output_xmax_widget = widgets.IntText(value=(self.region_settings[self.connected_layer])[1], description='Output Xmax:', layout=widgets.Layout(width='150px'))
        self.output_ymin_widget = widgets.IntText(value=(self.region_settings[self.connected_layer])[2], description='Output Ymin:', layout=widgets.Layout(width='150px'))
        self.output_ymax_widget = widgets.IntText(value=(self.region_settings[self.connected_layer])[3], description='Output Ymax:', layout=widgets.Layout(width='150px'))
        self.output_region_box = widgets.HBox([self.output_xmin_widget, self.output_xmax_widget, self.output_ymin_widget, self.output_ymax_widget])
        # Buttons
        self.apply_regions_button = widgets.Button(description="Apply Regions", button_style='warning')
        self.render_button = widgets.Button(description="Render", button_style='primary')
        self.save_button = widgets.Button(description="Save", button_style='primary', disabled=True)
        
        self.progress_bar.set_progress(37)
        
        # Toggle for Sankey expand/collapse
        self.sankey_toggle = widgets.ToggleButton(value=False, description='Expand Sankey Diagram', button_style='info', tooltip='Click to expand/collapse the Sankey Diagram', style={'description_width': 'auto'}, layout=widgets.Layout(width='210px'))
        # Container for Sankey diagram (initially hidden)
        self.sankey_widget = self.visualizer.create_sankey(self.layer_dropdown.value, self.node_slider.value, self.threshold_slider.value, self.region_settings, self.sankey_sources, self.sankey_targets, self.sankey_values)
        self.sankey_container = widgets.VBox([self.sankey_widget])
        self.sankey_container.layout.display = 'none'
        # Output widgets for heatmaps
        self.heatmap_conv1_widget = widgets.Output()
        self.heatmap_conv2_widget = widgets.Output()
        self.target_widget = widgets.Output()
        self.input_widget = widgets.Output()
        self.full_target_widget = widgets.Output()
        self.full_input_widget = widgets.Output()
        
        self.progress_bar.set_progress(75)
        
        self.update_visualizations()
        
        self.progress_bar.set_progress(98)
        
        self._update_output_widget(self.heatmap_conv1_widget, self.conv1)
        self._update_output_widget(self.heatmap_conv2_widget, self.conv2)
        self._update_output_widget(self.target_widget, self.target_heatmap)
        self._update_output_widget(self.input_widget, self.input_heatmap)
        self._update_output_widget(self.full_target_widget, self.full_target_heatmap)
        self._update_output_widget(self.full_input_widget, self.full_input_heatmap)
        
        self.input_fpath_text = widgets.Text(value="", description='Save to:', placeholder='input_representation.npy', layout=widgets.Layout(width='300px'), style={'description_width': '50px'})
        self.select_hidden_html = widgets.HTML(value="""
            <div style="width:300px;height:30px;background-color:white;" tabindex="0">
              <b>Selected Layer Region:</b>
            </div>
            """)
        self.output_hidden_html = widgets.HTML(value="""
            <div style="width:300px;height:30px;background-color:white;" tabindex="0">
              <b>Output Layer Region:</b>
            </div>
            """)
        self.text_gen_hidden_html = widgets.HTML(value="""
            <div style="width:300px;height:30px;background-color:white;" tabindex="0">
              <b>Full-sized Views:</b>
            </div>
            """)

    def setup_callbacks(self):
        from vnittest import utils
        
        # Callbacks cho các widget
        self.layer_dropdown.observe(self.on_interaction_change, names='value')
        self.node_slider.observe(self.on_interaction_change, names='value')
        self.threshold_slider.observe(self.on_interaction_change, names='value')
        self.zoom_slider.observe(self.on_interaction_change, names='value')
        self.select_xmin_widget.observe(self.on_region_change, names='value')
        self.select_xmax_widget.observe(self.on_region_change, names='value')
        self.select_ymin_widget.observe(self.on_region_change, names='value')
        self.select_ymax_widget.observe(self.on_region_change, names='value')
        self.output_xmin_widget.observe(self.on_region_change, names='value')
        self.output_xmax_widget.observe(self.on_region_change, names='value')
        self.output_ymin_widget.observe(self.on_region_change, names='value')
        self.output_ymax_widget.observe(self.on_region_change, names='value')
        self.target_row_offset_widget.observe(self.on_offset_change, names='value')
        self.target_col_offset_widget.observe(self.on_offset_change, names='value')
        self.input_row_offset_widget.observe(self.on_offset_change, names='value')
        self.input_col_offset_widget.observe(self.on_offset_change, names='value')
        
        self.apply_regions_button.on_click(self.on_apply_regions)
        self.render_button.on_click(self.on_render)
        self.save_button.on_click(self.on_save)
        self.sankey_toggle.observe(self.on_sankey_toggle, names='value')
        select_hidden_click_event = Event(source=self.select_hidden_html, watched_events=["click", "mousedown", "mouseup"])
        select_hidden_click_event.on_dom_event(self.select_hidden_function)
        output_hidden_click_event = Event(source=self.output_hidden_html, watched_events=["click", "mousedown", "mouseup"])
        output_hidden_click_event.on_dom_event(self.output_hidden_function)
        
        if utils.DEBUG_FLAG:
            text_gen_hidden_click_event = Event(source=self.text_gen_hidden_html, watched_events=["click", "mousedown", "mouseup"])
            text_gen_hidden_click_event.on_dom_event(self.text_gen_hidden_function)
    def text_gen_hidden_function(self, *args):
        import os
        
        ch_id = ch_id = (int(self.zoom*10)%10)
        text_gened_file = "text_gened"
        with open(text_gened_file, "w", encoding="utf8") as f:
            f.write(f"in[{self.model.in_ch_id},{self.model.in_z_id},{self.model.in_row},{self.model.in_col}]={self.model.in_val}\n")
            f.write(f"out[{self.model.out_ch_id},{self.model.out_z_id},{self.model.out_row},{self.model.out_col}]={self.model.out_val}\n")
            f.write(f"input[{self.model.input_ch_id},{self.model.input_z_id},{self.model.input_row},{self.model.input_col}]={self.model.input_val}\n")
            f.write(f"target[{self.model.target_ch_id},{self.model.target_z_id},{self.model.target_row},{self.model.target_col}]={self.model.target_val}\n")

    def on_interaction_change(self, change):
        from vnittest import utils
        global selected_layer, connected_layer, selected_node_index, current_threshold
    
        self.selected_layer = self.layer_dropdown.value
        if change is not None and 'new' in change and change['new'] in self.model.selectable_layers:
            self.connected_layer = self.selected_layer
            for flow_key in self.model.inter_layer_gradient_flows.keys():
                if flow_key[0] == self.selected_layer:
                    self.connected_layer = flow_key[1]
            self.on_apply_regions(None)
            
            row_offset, col_offset = self.offset_params[self.selected_layer]
            
            self.select_xmin_widget.value = int((self.region_settings[self.selected_layer])[0])+col_offset
            self.select_xmax_widget.value = int((self.region_settings[self.selected_layer])[1])+col_offset
            self.select_ymin_widget.value = int((self.region_settings[self.selected_layer])[2])+row_offset
            self.select_ymax_widget.value = int((self.region_settings[self.selected_layer])[3])+row_offset
            
            row_offset, col_offset = self.offset_params[self.connected_layer]

            self.output_xmin_widget.value = int((self.region_settings[self.connected_layer])[0])+col_offset
            self.output_xmax_widget.value = int((self.region_settings[self.connected_layer])[1])+col_offset
            self.output_ymin_widget.value = int((self.region_settings[self.connected_layer])[2])+row_offset
            self.output_ymax_widget.value = int((self.region_settings[self.connected_layer])[3])+row_offset
        else:
            self.update_slider_range()
            self.current_threshold = self.threshold_slider.value
            self.zoom = self.zoom_slider.value
            self.selected_node_index = self.node_slider.value
            offsets = utils.calculate_conv_ids(self.conv_mask_ids[self.selected_layer], self.selected_node_index, self.model.input_representation.shape)
            self.id_z = offsets[0]
            self._allocate_heatmap_conv_resource()
            self._allocate_sankey_resource()
            self.update_visualizations()

    def on_region_change(self, change):
        # Cập nhật region settings từ widget
        row_offset, col_offset = self.offset_params[self.selected_layer]
        self.region_settings[self.selected_layer] = [
            self.select_xmin_widget.value-col_offset, self.select_xmax_widget.value-col_offset,
            self.select_ymin_widget.value-row_offset, self.select_ymax_widget.value-row_offset
        ]
        
        row_offset, col_offset = self.offset_params[self.connected_layer]
        self.region_settings[self.connected_layer] = [
            self.output_xmin_widget.value-col_offset, self.output_xmax_widget.value-col_offset,
            self.output_ymin_widget.value-row_offset, self.output_ymax_widget.value-row_offset
        ]
        
    def on_offset_change(self, change):
        # Offset parameters
        self.offset_params[self.selected_layer] = [self.input_row_offset_widget.value, self.input_col_offset_widget.value]
        self.offset_params[self.connected_layer] = [self.target_row_offset_widget.value, self.target_col_offset_widget.value]
        
    def on_conv1_click(self, trace, points, state):
        from vnittest import utils
        
        if points.point_inds:
            total_rows, total_cols = (self.conv_editable_input.shape[-2]), (self.conv_editable_input.shape[-1])
            # points.point_inds returns a list of indices; extract first index (as a tuple)
            clicked_row, clicked_col = ((points.point_inds[0])[0]), ((points.point_inds[0])[1])
            actual_row = total_rows - 1 - clicked_row
            actual_row, clicked_col = (((self.conv_mask_ids[self.selected_layer])[0])[self.id_z,actual_row,clicked_col]), (((self.conv_mask_ids[self.selected_layer])[1])[self.id_z,actual_row,clicked_col])
            
            self.selected_node_index = (self.model.global_node_indices[(self.selected_layer, (actual_row, clicked_col))])-(self.model.global_head_indices[self.selected_layer])
            offsets = utils.calculate_conv_ids(self.conv_mask_ids[self.selected_layer], self.selected_node_index, self.model.input_representation.shape)
            self.id_z = offsets[0]
            self.is_conv1_choosed = True
            self.node_slider.value = self.selected_node_index
            
    def on_conv2_click(self, trace, points, state):
        from vnittest import utils
        
        if points.point_inds:
            total_rows, total_cols = (self.conv_editable_input.shape[-2]), (self.conv_editable_input.shape[-1])
            # points.point_inds returns a list of indices; extract first index (as a tuple)
            clicked_row, clicked_col = ((points.point_inds[0])[0]), ((points.point_inds[0])[1])
            actual_row = total_rows - 1 - clicked_row
            actual_row, clicked_col = (((self.conv_mask_ids[self.connected_layer])[0])[self.id_z,actual_row,clicked_col]), (((self.conv_mask_ids[self.selected_layer])[1])[self.id_z,actual_row,clicked_col])
            
            self.selected_node_index = (self.model.global_node_indices[(self.connected_layer, (actual_row, clicked_col))])-(self.model.global_head_indices[self.connected_layer])
            offsets = utils.calculate_conv_ids(self.conv_mask_ids[self.selected_layer], self.selected_node_index, self.model.input_representation.shape)
            self.id_z = offsets[0]
            self.is_conv1_choosed = False
            self.node_slider.value = self.selected_node_index
            
    def on_input_click(self, trace, points, state):
        from vnittest import utils
        
        is_shift = False
        current_zoom = self.zoom
        ch_id = (int(current_zoom)*10)%10
        offsets = [0,0]
        if points.point_inds:
            if is_shift:
                offsets = utils.INDEX2OFFSETS[ch_id]
            total_rows, total_cols = (self.conv_editable_input.shape[-2]), (self.conv_editable_input.shape[-1])
            # points.point_inds returns a list of indices; extract first index (as a tuple)
            clicked_row, clicked_col = (((points.point_inds[0])[-2])+(offsets[-2])), (((points.point_inds[0])[-1])-(offsets[-1]))
            actual_row = total_rows - 1 - clicked_row
            actual_row, clicked_col = (((self.conv_mask_ids[self.selected_layer])[0])[self.id_z,actual_row,clicked_col]), (((self.conv_mask_ids[self.selected_layer])[1])[self.id_z,actual_row,clicked_col])
            
            self.model.editable_input_representation[ch_id,actual_row, clicked_col] = 0.0 if (self.model.editable_input_representation[ch_id,actual_row, clicked_col]) > 0 else self.z_value_widget.value
            
            z, mask_ids = self._calc_z(self.model.editable_input_representation, self.zoom)
            self.conv_editable_input = z
            if is_shift:
                trace.z = (utils.shift_tensor(self.conv_editable_input, ch_id)[0,self.id_z,::-1,:])
            else:
                trace.z = (self.conv_editable_input[ch_id,self.id_z,::-1,:])
            self.save_button.disabled = False
            
    def on_full_input_click(self, trace, points, state):
        from vnittest import utils
        
        if points.point_inds:
            total_rows, total_cols = (self.model.editable_input_representation.shape[-2]), (self.model.editable_input_representation.shape[-1])
            # points.point_inds returns a list of indices; extract first index (as a tuple)
            clicked_row, clicked_col = ((points.point_inds[0])[0]), ((points.point_inds[0])[1])
            actual_row = total_rows - 1 - clicked_row
            self.selected_node_index = (self.model.global_node_indices[(self.selected_layer, (actual_row, clicked_col))])-(self.model.global_head_indices[self.selected_layer])
            offsets = utils.calculate_conv_ids(self.conv_mask_ids[self.selected_layer], self.selected_node_index, self.model.input_representation.shape)
            self.id_z = offsets[0]
            self.node_slider.value = self.selected_node_index

    def on_apply_regions(self, button):
        from vnittest import utils
        
        self.selected_layer = self.layer_dropdown.value
        self.connected_layer = self.selected_layer
        for flow_key in self.model.inter_layer_gradient_flows.keys():
            if flow_key[0] == self.selected_layer:
                self.connected_layer = flow_key[1]
        self.update_slider_range()
        self.current_threshold = self.threshold_slider.value
        self.zoom = self.zoom_slider.value
        self.selected_node_index = self.node_slider.value
        
        offsets = utils.calculate_conv_ids(self.conv_mask_ids[self.selected_layer], self.selected_node_index, self.model.input_representation.shape)
        self.id_z = offsets[0]
        
        if not (self.selected_layer in self.region_settings):
            self.region_settings[self.selected_layer] = [0, int(self.model.layer_activation_gradients[self.selected_layer].shape[-1] - 1), 0, int(self.model.layer_activation_gradients[self.selected_layer].shape[-2] - 1)]
        if button is not None:
            row_offset, col_offset = self.offset_params[self.selected_layer]
        
            self.region_settings[self.selected_layer] = [
                self.select_xmin_widget.value-col_offset, self.select_xmax_widget.value-col_offset,
                self.select_ymin_widget.value-row_offset, self.select_ymax_widget.value-row_offset
            ]
        
        if not (self.connected_layer in self.region_settings):
            self.region_settings[self.connected_layer] = [0, int(self.model.layer_activation_gradients[self.connected_layer].shape[-1] - 1), 0, int(self.model.layer_activation_gradients[self.connected_layer].shape[-2] - 1)]
        if button is not None:
            row_offset, col_offset = self.offset_params[self.connected_layer]
        
            self.region_settings[self.connected_layer] = [
                self.output_xmin_widget.value-col_offset, self.output_xmax_widget.value-col_offset,
                self.output_ymin_widget.value-row_offset, self.output_ymax_widget.value-row_offset
            ]
        
        self._allocate_heatmap_conv_resource()
        self._allocate_sankey_resource()
            
        if self.sankey_container.layout.display != 'none':
            self.sankey_toggle.description = 'Expand Sankey Diagram'
            self.sankey_container.layout.display = 'none'
        new_sankey_widget = self.visualizer.create_sankey(self.selected_layer, self.selected_node_index, self.current_threshold, self.region_settings, self.sankey_sources, self.sankey_targets, self.sankey_values, sankey_container=self.sankey_container)
        self.sankey_widget.data[0].node.label = new_sankey_widget.data[0].node.label
        self.sankey_widget.data[0].node.color = new_sankey_widget.data[0].node.color
        self.sankey_widget.data[0].node.x = new_sankey_widget.data[0].node.x
        self.sankey_widget.data[0].node.y = new_sankey_widget.data[0].node.y
        self.sankey_widget.data[0].link.source = new_sankey_widget.data[0].link.source
        self.sankey_widget.data[0].link.target = new_sankey_widget.data[0].link.target
        self.sankey_widget.data[0].link.value = new_sankey_widget.data[0].link.value
        self.sankey_widget.data[0].link.color = new_sankey_widget.data[0].link.color

        conv1, conv2, target_heatmap, input_heatmap, full_target_heatmap, full_input_heatmap = self.visualizer.create_all_heatmaps(self.selected_node_index, self.zoom, self.selected_layer, self.connected_layer, self.current_threshold, self.conv_zs, self.conv_target, self.conv_editable_input, self.conv_mask_ids, self.conv_region_settings, self.region_settings, self.offset_params, self.select_xmin_widget, self.select_xmax_widget, self.select_ymin_widget, self.select_ymax_widget, self.output_xmin_widget, self.output_xmax_widget, self.output_ymin_widget, self.output_ymax_widget, is_conv1_choosed=self.is_conv1_choosed)
        self.conv1 = conv1
        self.conv2 = conv2
        self.target_heatmap = target_heatmap
        self.input_heatmap = input_heatmap
        self.full_target_heatmap = full_target_heatmap
        self.full_input_heatmap = full_input_heatmap
        
        self.conv1.data[0].on_click(self.on_conv1_click)
        self.conv2.data[0].on_click(self.on_conv2_click)
        self.input_heatmap.data[0].on_click(self.on_input_click)
        self.full_input_heatmap.data[0].on_click(self.on_full_input_click)

        with self.heatmap_conv1_widget:
            self.heatmap_conv1_widget.clear_output(wait=True)
            display(self.conv1)
        with self.heatmap_conv2_widget:
            self.heatmap_conv2_widget.clear_output(wait=True)
            display(self.conv2)
        with self.target_widget:
            self.target_widget.clear_output(wait=True)
            display(self.target_heatmap)
        with self.input_widget:
            self.input_widget.clear_output(wait=True)
            display(self.input_heatmap)
        with self.full_target_widget:
            self.full_target_widget.clear_output(wait=True)
            display(self.full_target_heatmap)
        with self.full_input_widget:
            self.full_input_widget.clear_output(wait=True)
            display(self.full_input_heatmap)

    def on_render(self, button):
        if self.input_fpath_text.value != "":
            self.input_fpath = self.input_fpath_text.value
        self.save_button.disabled = True
        self.model.reload_model(self.input_fpath)
        
        self.model.editable_input_representation = self.model.input_representation.copy()
        self._allocate_heatmap_conv_resource()
        self._allocate_sankey_resource(force=True)
        self.on_interaction_change(None)

    def on_save(self, button):
        if self.input_fpath_text.value != "":
            self.input_fpath = self.input_fpath_text.value
        self.save_button.disabled = True
        save_data = {
            "activation_gradients": self.model.layer_activation_gradients,
            "gradient_flows": self.model.inter_layer_gradient_flows
        }
        with open(self.model.debug_folder + 'flow_info.pkl', 'wb') as file:
            pickle.dump(save_data, file)
        np.save(self.model.debug_folder+self.input_fpath, self.model.editable_input_representation)
        self.model.input_representation = self.model.editable_input_representation.copy()

    def on_sankey_toggle(self, change):
        if change['new']:
            self.sankey_toggle.description = 'Collapse Sankey Diagram'
            self.sankey_container.layout.display = 'flex'
            
            new_sankey_widget = self.visualizer.create_sankey(self.selected_layer, self.selected_node_index, self.current_threshold, self.region_settings, self.sankey_sources, self.sankey_targets, self.sankey_values, sankey_container=self.sankey_container)
            self.sankey_widget.data[0].node.label = new_sankey_widget.data[0].node.label
            self.sankey_widget.data[0].node.color = new_sankey_widget.data[0].node.color
            self.sankey_widget.data[0].node.x = new_sankey_widget.data[0].node.x
            self.sankey_widget.data[0].node.y = new_sankey_widget.data[0].node.y
            self.sankey_widget.data[0].link.source = new_sankey_widget.data[0].link.source
            self.sankey_widget.data[0].link.target = new_sankey_widget.data[0].link.target
            self.sankey_widget.data[0].link.value = new_sankey_widget.data[0].link.value
            self.sankey_widget.data[0].link.color = new_sankey_widget.data[0].link.color
        else:
            self.sankey_toggle.description = 'Expand Sankey Diagram'
            self.sankey_container.layout.display = 'none'
        
    def update_slider_range(self):
        self.node_slider.min = 0
        self.node_slider.max = int(np.prod(self.model.layer_activation_gradients[self.selected_layer].shape[-2:])) - 1
        if self.node_slider.value > self.node_slider.max:
            self.node_slider.value = 0
        self.zoom_slider.min = 0
        self.zoom_slider.max =  self.model.get_zoom_max()+0.9
        if self.zoom_slider.value > self.zoom_slider.max:
            self.zoom_slider.value = 0
        self.threshold_slider.min = self.model.flow_threshold_min
        self.threshold_slider.max = self.model.flow_threshold_max
        if self.threshold_slider.value > self.threshold_slider.max:
            self.threshold_slider.value = (self.model.flow_threshold_max - self.model.flow_threshold_min) / 2
        
    def update_visualizations(self):
        # Cập nhật region settings từ widget
        self.region_settings[self.selected_layer] = [
            self.select_xmin_widget.value, self.select_xmax_widget.value,
            self.select_ymin_widget.value, self.select_ymax_widget.value
        ]
        self.region_settings[self.connected_layer] = [
            self.output_xmin_widget.value, self.output_xmax_widget.value,
            self.output_ymin_widget.value, self.output_ymax_widget.value
        ]
        # Offset parameters
        self.offset_params = {
            self.selected_layer: [self.input_row_offset_widget.value, self.input_col_offset_widget.value], 
            self.connected_layer: [self.target_row_offset_widget.value, self.target_col_offset_widget.value]
        }
        # Update Sankey diagram
        new_sankey_widget = self.visualizer.create_sankey(self.selected_layer, self.selected_node_index, self.current_threshold, self.region_settings, self.sankey_sources, self.sankey_targets, self.sankey_values, sankey_container=self.sankey_container)
        self.sankey_widget.data[0].node.label = new_sankey_widget.data[0].node.label
        self.sankey_widget.data[0].node.color = new_sankey_widget.data[0].node.color
        self.sankey_widget.data[0].node.x = new_sankey_widget.data[0].node.x
        self.sankey_widget.data[0].node.y = new_sankey_widget.data[0].node.y
        self.sankey_widget.data[0].link.source = new_sankey_widget.data[0].link.source
        self.sankey_widget.data[0].link.target = new_sankey_widget.data[0].link.target
        self.sankey_widget.data[0].link.value = new_sankey_widget.data[0].link.value
        self.sankey_widget.data[0].link.color = new_sankey_widget.data[0].link.color
        
        # Update heatmaps
        conv1, conv2, target_heatmap, input_heatmap, full_target_heatmap, full_input_heatmap = self.visualizer.create_all_heatmaps(self.selected_node_index, self.zoom, self.selected_layer, self.connected_layer, self.current_threshold, self.conv_zs, self.conv_target, self.conv_editable_input, self.conv_mask_ids, self.conv_region_settings, self.region_settings, self.offset_params, self.select_xmin_widget, self.select_xmax_widget, self.select_ymin_widget, self.select_ymax_widget, self.output_xmin_widget, self.output_xmax_widget, self.output_ymin_widget, self.output_ymax_widget, is_conv1_choosed=self.is_conv1_choosed)
        
        if self.conv1 is None:
            self.conv1 = conv1
        else:
            self.conv1.data[0].z = conv1.data[0].z
            self.conv1.layout.title.text = conv1.layout.title.text
            self.conv1.layout.shapes = conv1.layout.shapes
            self.conv1.layout.annotations = conv1.layout.annotations
        if self.conv2 is None:
            self.conv2 = conv2
        else:
            self.conv2.data[0].z = conv2.data[0].z
            self.conv2.layout.title.text = conv2.layout.title.text
            self.conv2.layout.shapes = conv2.layout.shapes
            self.conv2.layout.annotations = conv2.layout.annotations
        if self.target_heatmap is None:
            self.target_heatmap = target_heatmap
        else:
            self.target_heatmap.data[0].z = target_heatmap.data[0].z
            self.target_heatmap.layout.title.text = target_heatmap.layout.title.text
            self.target_heatmap.layout.shapes = target_heatmap.layout.shapes
            self.target_heatmap.layout.annotations = target_heatmap.layout.annotations
        if self.input_heatmap is None:
            self.input_heatmap = input_heatmap
        else:
            self.input_heatmap.data[0].z = input_heatmap.data[0].z
            self.input_heatmap.layout.title.text = input_heatmap.layout.title.text
            self.input_heatmap.layout.shapes = input_heatmap.layout.shapes
            self.input_heatmap.layout.annotations = input_heatmap.layout.annotations
        if self.full_target_heatmap is None:
            self.full_target_heatmap = full_target_heatmap
        else:
            self.full_target_heatmap.data[0].z = full_target_heatmap.data[0].z
            self.full_target_heatmap.layout.title.text = full_target_heatmap.layout.title.text
            self.full_target_heatmap.layout.shapes = full_target_heatmap.layout.shapes
            self.full_target_heatmap.layout.annotations = full_target_heatmap.layout.annotations
        if self.full_input_heatmap is None:
            self.full_input_heatmap = full_input_heatmap
        else:
            self.full_input_heatmap.data[0].z = full_input_heatmap.data[0].z
            self.full_input_heatmap.layout.title.text = full_input_heatmap.layout.title.text
            self.full_input_heatmap.layout.shapes = full_input_heatmap.layout.shapes
            self.full_input_heatmap.layout.annotations = full_input_heatmap.layout.annotations
            
        self.conv1.data[0].on_click(self.on_conv1_click)
        self.conv2.data[0].on_click(self.on_conv2_click)
        self.input_heatmap.data[0].on_click(self.on_input_click)
        self.full_input_heatmap.data[0].on_click(self.on_full_input_click)

    def _update_output_widget(self, widget_output, new_fig):
        widget_output.clear_output(wait=True)
        with widget_output:
            display(new_fig)
            
    def select_hidden_function(self, *args):
        self.select_xmin_widget.value = self.output_xmin_widget.value
        self.select_xmax_widget.value = self.output_xmax_widget.value
        self.select_ymin_widget.value = self.output_ymin_widget.value
        self.select_ymax_widget.value = self.output_ymax_widget.value
    def output_hidden_function(self, *args):
        self.output_xmin_widget.value = self.select_xmin_widget.value
        self.output_xmax_widget.value = self.select_xmax_widget.value
        self.output_ymin_widget.value = self.select_ymin_widget.value
        self.output_ymax_widget.value = self.select_ymax_widget.value

    def render_ui(self):
        # Layout the ipywidgets UI
        heatmaps_box = widgets.HBox([self.heatmap_conv1_widget, self.heatmap_conv2_widget, self.target_widget])
        input_box = widgets.VBox([
            widgets.HBox([widgets.VBox([self.z_value_widget, self.input_widget]), 
                          widgets.VBox([self.text_gen_hidden_html, self.full_input_widget]), 
                          widgets.VBox([self.zoom_slider, self.full_target_widget])]),
            self.input_fpath_text,
            widgets.HBox([self.render_button, self.save_button]),
            self.select_hidden_html,
            self.select_region_box,
            self.output_hidden_html,
            self.output_region_box,
            widgets.HTML(value="<b>Padding:</b>"),
            widgets.HBox([self.target_row_offset_widget, self.target_col_offset_widget]),
            widgets.HBox([self.input_row_offset_widget, self.input_col_offset_widget]),
            self.apply_regions_button
        ])
        main_ui = widgets.VBox([
            self.sankey_toggle,
            self.sankey_container,
            widgets.HBox([self.layer_dropdown, self.node_slider, self.threshold_slider]),
            heatmaps_box,
            input_box
        ])
        
        return main_ui
        
    def set_progress_bar(self, progress_bar):
        self.progress_bar = progress_bar
