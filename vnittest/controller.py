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
        self.allocate_sankey_resource()
        self.progress_bar.set_progress(25)
        
        self.selected_layer = self.model.selectable_layers[0]
        self.connected_layer = self.selected_layer
        for flow_key in self.model.inter_layer_gradient_flows.keys():
            if flow_key[0] == self.selected_layer:
                self.connected_layer = flow_key[1]
        self.selected_node_index = 0
        self.current_threshold = (self.model.flow_threshold_max - self.model.flow_threshold_min) / 2

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
        # Offset parameters for target and input representations
        self.offset_params = {
            'target_row': 0,
            'target_col': 0,
            'input_row': 0,
            'input_col': 0
        }
        
        self.conv1, self.conv2, self.target_heatmap, self.input_heatmap = None, None, None, None
        
        self.build_widgets()
        self.setup_callbacks()
        
    def allocate_sankey_resource(self):
        # Build lists for Sankey diagram: sources, targets, and flow values
        self.sankey_sources, self.sankey_targets, self.sankey_values = [], [], []
        for layer_pair, flow_matrix in self.model.flow_matrices.items():
            source_layer, target_layer = layer_pair
            source_positions = self.model.layer_node_positions[source_layer]
            target_positions = self.model.layer_node_positions[target_layer]
            flow_value = 0
            if flow_matrix is not None:
                if isinstance(flow_matrix, dict):
                    indices_in = (flow_matrix["indices_in"])
                    indices_out = (flow_matrix["indices_out"])
                    id_in = 0
                    id_out = 0
                    for i, (r1, c1) in enumerate(source_positions):
                        for j, (r2, c2) in enumerate(target_positions):
                            if flow_value > 0:
                                while id_in<len(indices_in) and ((indices_in[0])[id_in]) < r1 and ((indices_in[1])[id_in]) < c1:
                                    id_in += 1
                                while id_out<len(indices_out) and ((indices_out[0])[id_out]) < r2 and ((indices_out[1])[id_out]) < c2:
                                    id_out += 1
                                if (id_in<len(indices_in) and ((indices_in[0])[id_in]) == r1 and ((indices_in[1])[id_in]) == c1) or (id_out<len(indices_out) and ((indices_out[0])[id_out]) == r2 and ((indices_out[1])[id_out]) == c2):
                                    self.sankey_sources.append(self.model.global_node_indices[(source_layer, (r1, c1))])
                                    self.sankey_targets.append(self.model.global_node_indices[(target_layer, (r2, c2))])
                                    self.sankey_values.append(1)
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
        self.sankey_toggle = widgets.ToggleButton(value=False, description='Expand Sankey Diagram', button_style='info', tooltip='Click to expand/collapse the Sankey Diagram')
        # Container for Sankey diagram (initially hidden)
        self.sankey_widget = self.visualizer.create_sankey(self.layer_dropdown.value, self.node_slider.value, self.threshold_slider.value, self.region_settings, self.sankey_sources, self.sankey_targets, self.sankey_values)
        self.sankey_container = widgets.VBox([self.sankey_widget])
        self.sankey_container.layout.display = 'none'
        # Output widgets for heatmaps
        self.heatmap_conv1_widget = widgets.Output()
        self.heatmap_conv2_widget = widgets.Output()
        self.target_widget = widgets.Output()
        self.input_widget = widgets.Output()
        
        self.progress_bar.set_progress(75)
        
        self.update_visualizations()
        
        self.progress_bar.set_progress(98)
        
        self._update_output_widget(self.heatmap_conv1_widget, self.conv1)
        self._update_output_widget(self.heatmap_conv2_widget, self.conv2)
        self._update_output_widget(self.target_widget, self.target_heatmap)
        self._update_output_widget(self.input_widget, self.input_heatmap)
        
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

    def setup_callbacks(self):
        # Callbacks cho các widget
        self.layer_dropdown.observe(self.on_interaction_change, names='value')
        self.node_slider.observe(self.on_interaction_change, names='value')
        self.threshold_slider.observe(self.on_interaction_change, names='value')
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

    def on_interaction_change(self, change):
        global selected_layer, connected_layer, selected_node_index, current_threshold
    
        self.selected_layer = self.layer_dropdown.value
        if change is not None and 'new' in change and change['new'] in self.model.selectable_layers:
            self.connected_layer = self.selected_layer
            for flow_key in self.model.inter_layer_gradient_flows.keys():
                if flow_key[0] == self.selected_layer:
                    self.connected_layer = flow_key[1]
            self.on_apply_regions(None)
            
            self.select_xmin_widget.value = int((self.region_settings[self.selected_layer])[0])
            self.select_xmax_widget.value = int((self.region_settings[self.selected_layer])[1])
            self.select_ymin_widget.value = int((self.region_settings[self.selected_layer])[2])
            self.select_ymax_widget.value = int((self.region_settings[self.selected_layer])[3])

            self.output_xmin_widget.value = int((self.region_settings[self.connected_layer])[0])
            self.output_xmax_widget.value = int((self.region_settings[self.connected_layer])[1])
            self.output_ymin_widget.value = int((self.region_settings[self.connected_layer])[2])
            self.output_ymax_widget.value = int((self.region_settings[self.connected_layer])[3])
        else:
            self.update_slider_range()
            self.current_threshold = self.threshold_slider.value
            self.selected_node_index = self.node_slider.value
            self.update_visualizations()

    def on_region_change(self, change):
        # Cập nhật region settings từ widget
        self.region_settings[self.selected_layer] = [
            self.select_xmin_widget.value, self.select_xmax_widget.value,
            self.select_ymin_widget.value, self.select_ymax_widget.value
        ]
        self.region_settings[self.connected_layer] = [
            self.output_xmin_widget.value, self.output_xmax_widget.value,
            self.output_ymin_widget.value, self.output_ymax_widget.value
        ]
        
    def on_offset_change(self, change):
        # Offset parameters
        self.offset_params = {
            'target_row': self.target_row_offset_widget.value,
            'target_col': self.target_col_offset_widget.value,
            'input_row': self.input_row_offset_widget.value,
            'input_col': self.input_col_offset_widget.value
        }
        
    def on_conv1_click(self, trace, points, state):
        if points.point_inds:
            total_rows, total_cols = self.model.editable_input_representation.shape
            # points.point_inds returns a list of indices; extract first index (as a tuple)
            clicked_row_display = (points.point_inds[0])[0]
            clicked_col = (points.point_inds[0])[1]
            actual_row = total_rows - 1 - clicked_row_display
            self.selected_node_index = self.model.global_node_indices[(self.selected_layer, (actual_row, clicked_col))]
            self.node_slider.value = self.selected_node_index
            
    def on_input_click(self, trace, points, state):
        if points.point_inds:
            total_rows, total_cols = self.model.editable_input_representation.shape
            # points.point_inds returns a list of indices; extract first index (as a tuple)
            clicked_row_display = (points.point_inds[0])[0]
            clicked_col = (points.point_inds[0])[1]
            actual_row = total_rows - 1 - clicked_row_display
            self.model.editable_input_representation[actual_row, clicked_col] = 0.0 if self.model.editable_input_representation[actual_row, clicked_col] > 0 else self.z_value_widget.value
            trace.z = self.model.editable_input_representation[::-1, :]
            self.save_button.disabled = False

    def on_apply_regions(self, button):
        self.selected_layer = self.layer_dropdown.value
        self.connected_layer = self.selected_layer
        for flow_key in self.model.inter_layer_gradient_flows.keys():
            if flow_key[0] == self.selected_layer:
                self.connected_layer = flow_key[1]
        self.update_slider_range()
        self.current_threshold = self.threshold_slider.value
        self.selected_node_index = self.node_slider.value
        
        if not (self.selected_layer in self.region_settings):
            self.region_settings[self.selected_layer] = [0, int(self.model.layer_activation_gradients[self.selected_layer].shape[-1] - 1), 0, int(self.model.layer_activation_gradients[self.selected_layer].shape[-2] - 1)]
        if button is not None:
            self.region_settings[self.selected_layer] = (self.select_xmin_widget.value, self.select_xmax_widget.value, self.select_ymin_widget.value, self.select_ymax_widget.value)
        
        if not (self.connected_layer in self.region_settings):
            self.region_settings[self.connected_layer] = [0, int(self.model.layer_activation_gradients[self.connected_layer].shape[-1] - 1), 0, int(self.model.layer_activation_gradients[self.connected_layer].shape[-2] - 1)]
        if button is not None:
            self.region_settings[self.connected_layer] = (self.output_xmin_widget.value, self.output_xmax_widget.value, self.output_ymin_widget.value, self.output_ymax_widget.value)

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

        conv1, conv2, target_heatmap, input_heatmap = self.visualizer.create_all_heatmaps(self.selected_node_index, self.selected_layer, self.connected_layer, self.current_threshold, self.offset_params, self.region_settings, 
                                                                                          self.select_xmin_widget, self.select_xmax_widget, self.select_ymin_widget, self.select_ymax_widget, 
                                                                                          self.output_xmin_widget, self.output_xmax_widget, self.output_ymin_widget, self.output_ymax_widget)
        self.conv1 = conv1
        self.conv2 = conv2
        self.target_heatmap = target_heatmap
        self.input_heatmap = input_heatmap
        
        self.conv1.data[0].on_click(self.on_conv1_click)
        self.input_heatmap.data[0].on_click(self.on_input_click)

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

    def on_render(self, button):
        if self.input_fpath_text.value != "":
            self.input_fpath = self.input_fpath_text.value
        self.save_button.disabled = True
        self.model.reload_data(self.input_fpath)
        
        self.model.editable_input_representation = self.model.input_representation.copy()
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
            'target_row': self.target_row_offset_widget.value,
            'target_col': self.target_col_offset_widget.value,
            'input_row': self.input_row_offset_widget.value,
            'input_col': self.input_col_offset_widget.value
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
        conv1, conv2, target_heatmap, input_heatmap = self.visualizer.create_all_heatmaps(
            self.selected_node_index, self.selected_layer, self.connected_layer, self.current_threshold, self.offset_params, self.region_settings, 
            self.select_xmin_widget, self.select_xmax_widget, self.select_ymin_widget, self.select_ymax_widget, 
            self.output_xmin_widget, self.output_xmax_widget, self.output_ymin_widget, self.output_ymax_widget
        )
        
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
            
        self.conv1.data[0].on_click(self.on_conv1_click)
        self.input_heatmap.data[0].on_click(self.on_input_click)

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
            self.input_widget,
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
        display(main_ui)
        
    def set_progress_bar(self, progress_bar):
        self.progress_bar = progress_bar