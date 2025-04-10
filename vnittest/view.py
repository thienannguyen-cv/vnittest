# ======================================================
# View: Visualization creation (Sankey and Heatmaps)
# ======================================================
from vnittest import GradientModel
import ipywidgets as widgets
import plotly.graph_objs as go

class ProgressBar(widgets.HTML):
    def __init__(self):
        super(ProgressBar, self).__init__()
        
    def set_progress(self, value):
        # Create a Progress with a description
        def create_progress_html(value, max_value=100, width_px=300, height_px=30):
            """
            Tạo đoạn HTML mô phỏng một thanh progress với phần trăm ở giữa.
            
            value:      giá trị hiện tại (số phần trăm).
            max_value:  giá trị tối đa (thường 100).
            width_px:   chiều rộng thanh progress.
            height_px:  chiều cao thanh progress.
            """
            # Tính % để hiển thị trong thanh màu
            percent_fill = (value / max_value) * 100

            # HTML cho thanh progress
            progress_html = f"""
            <div style="position: relative; width: {width_px}px; height: {height_px}px; background-color: #ddd;">
              <!-- Phần màu hiển thị progress -->
              <div style="
                  position: absolute; 
                  width: {percent_fill}%; 
                  height: 100%; 
                  background-color: #00bcd4;">
              </div>
              <!-- Phần text nằm chính giữa -->
              <div style="
                  position: absolute; 
                  width: 100%; 
                  text-align: center; 
                  line-height: {height_px}px; 
                  font-weight: bold;">
                {int(value)}%
              </div>
            </div>
            """
            return progress_html
        # Cập nhật HTML bên trong widget
        self.value = create_progress_html(value)

class Visualizer:
    def __init__(self, model: GradientModel):
        self.model = model
        
    def allocate_resource(self):
        self.allocate_sankey_resource()
        self.progress_bar.set_progress(25)
        
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
                        
    def create_progress_bar(self):
        # Tạo một widget HTML
        self.progress_bar = ProgressBar()
        
    def get_progress_bar(self):
        if (not hasattr(self, 'progress_bar')) or self.progress_bar is None:
            self.create_progress_bar()
        return self.progress_bar
        
    def set_progress_bar(self, progress_bar):
        self.progress_bar = progress_bar
    
    def display_progress_bar(self):
        display(self.progress_bar)
        
    def destroy_progress_bar(self):
        self.progress_bar.close()
    
    def create_sankey(self, current_layer, current_node_index, threshold_value, region_settings, sankey_container=None):
        new_node_labels, new_node_x, new_node_y, new_node_colors, new_link_colors = [], [], [], [], []
        new_sources, new_targets, new_values = [], [], []
        sankey_fig = None
        if sankey_container is None or sankey_container.layout.display == 'none':
            sankey_fig = go.FigureWidget(data=[go.Sankey(
                arrangement='fixed',
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=new_node_labels,
                    color=new_node_colors,
                    x=new_node_x,
                    y=new_node_y
                ),
                link=dict(
                    source=new_sources,
                    target=new_targets,
                    value=new_values,
                    color=new_link_colors
                )
            )])
        else:
            new_index_mapping = {}
            label_counter = 0
            for i, layer_label in enumerate(self.model.all_layers):
                H, W = self.model.layer_activation_gradients[layer_label].shape[-2:]
                self.model.layer_node_positions[layer_label] = [(r, c) for r in range(H) for c in range(W)]
                node_color_list = self.model.layer_color_maps[layer_label]
                updated_node_colors = []
                inner_counter = 0
                for j, (row, col) in enumerate(self.model.layer_node_positions[layer_label]):
                    global_index = self.model.global_node_indices[(layer_label, (row, col))]
                    if (layer_label not in region_settings) or ((region_settings[layer_label])[0] <= col <= (region_settings[layer_label])[1] and (region_settings[layer_label])[2] <= (H - 1 - row) <= (region_settings[layer_label])[3]):
                        new_node_labels.append(f"{layer_label}_{row},{col}")
                        new_node_x.append((float(i) + 0.01) / (len(self.model.all_layers) - 0.55))
                        if layer_label in region_settings:
                            denom = ((region_settings[layer_label])[1] - (region_settings[layer_label])[0] + 1) * ((region_settings[layer_label])[3] - (region_settings[layer_label])[2] + 1)
                            new_node_y.append((float(inner_counter) + 0.5) / denom)
                        else:
                            new_node_y.append((float(inner_counter) + 0.5) / (len(self.model.layer_node_positions[layer_label])))
                        updated_node_colors.append("red" if j == current_node_index else node_color_list[j])
                        new_index_mapping[global_index] = label_counter
                        inner_counter += 1
                        label_counter += 1
                new_node_colors += updated_node_colors

            for idx, src in enumerate(self.sankey_sources):
                if src in new_index_mapping and self.sankey_targets[idx] in new_index_mapping:
                    new_sources.append(new_index_mapping[src])
                    new_targets.append(new_index_mapping[self.sankey_targets[idx]])
                    new_values.append(self.sankey_values[idx])
                    if src == self.model.global_node_indices[(current_layer, self.model.layer_node_positions[current_layer][current_node_index])] and float(self.sankey_values[idx]) > threshold_value:
                        new_link_colors.append("red")
                    else:
                        new_link_colors.append(f"rgba({128*int(float(self.sankey_values[idx])>threshold_value)},"
                                                 f"{128*int(float(self.sankey_values[idx])>threshold_value)},"
                                                 f"{128*int(float(self.sankey_values[idx])>threshold_value)},"
                                                 f"{1.*int(float(self.sankey_values[idx])>threshold_value)+.1})")
            sankey_fig = go.FigureWidget(data=[go.Sankey(
                arrangement='fixed',
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=new_node_labels,
                    color=new_node_colors,
                    x=new_node_x,
                    y=new_node_y
                ),
                link=dict(
                    source=new_sources,
                    target=new_targets,
                    value=new_values,
                    color=new_link_colors
                )
            )])
            sankey_fig.update_layout(title_text='Gradient Flow', font_size=10)
        return sankey_fig

    def _create_heatmap_conv1(self, current_node_index, current_layer, region_settings, select_xmin_widget, select_xmax_widget, select_ymin_widget, select_ymax_widget):
        highlight_shapes = []
        grid_shape = self.model.layer_activation_gradients[current_layer].shape
        if current_node_index is not None:
            row, col = divmod(current_node_index, grid_shape[-1])
            highlight_shapes.append(dict(
                type='circle', xref='x', yref='y',
                x0=col - 0.5, y0=grid_shape[0] - 1 - row - 0.5,
                x1=col + 0.5, y1=grid_shape[0] - 1 - row + 0.5,
                line=dict(color='red', width=3)
            ))
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=(self.model.layer_activation_gradients[current_layer])[::-1, :],
            colorscale='Viridis',
            zmin=(self.model.layer_activation_gradients[current_layer]).min(),
            zmax=(self.model.layer_activation_gradients[current_layer]).max(),
            colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        ))
        try:
            xmin, xmax = ((region_settings[current_layer])[0]) - 0.5, ((region_settings[current_layer])[1]) + 0.5
            ymin, ymax = ((region_settings[current_layer])[2]) - 0.5, ((region_settings[current_layer])[3]) + 0.5
            heatmap_fig.update_layout(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax])
            )
        except Exception as e:
            pass

        def on_x_range_change(layout, x_range):
            select_xmin_widget.value = int(x_range[0])
            select_xmax_widget.value = int(x_range[1])
        def on_y_range_change(change, y_range):
            select_ymin_widget.value = int(y_range[0])
            select_ymax_widget.value = int(y_range[1])
        heatmap_fig.layout.on_change(on_x_range_change, 'xaxis.range')
        heatmap_fig.layout.on_change(on_y_range_change, 'yaxis.range')

        heatmap_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            width=300, height=300,
            xaxis=dict(dtick=1),
            yaxis=dict(dtick=1),
            shapes=highlight_shapes,
            annotations=[dict(
                text="Selected-Layer Gradient",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10)
            )]
        )
        return heatmap_fig

    def _create_heatmap_conv2(self, current_node_index, current_layer, connected_layer, threshold_value, region_settings, output_xmin_widget, output_xmax_widget, output_ymin_widget, output_ymax_widget):
        highlight_shapes = []

        grid_shape = self.model.layer_activation_gradients[connected_layer].shape
        if not isinstance(self.model.flow_matrices[(current_layer, connected_layer)], dict):
            flow_matrix_thresholded = None
            if self.model.flow_matrices[(current_layer, connected_layer)] is not None:
                flow_matrix_thresholded = ((self.model.flow_matrices[(current_layer, connected_layer)] > threshold_value)[current_node_index, :]).reshape(*grid_shape)
            for r in range(grid_shape[0]):
                for c in range(grid_shape[1]):
                    if (flow_matrix_thresholded is not None) and flow_matrix_thresholded[r, c] > 0:
                        highlight_shapes.append(dict(
                            type='circle', xref='x', yref='y',
                            x0=c - 0.5, y0=grid_shape[0] - 1 - r - 0.5,
                            x1=c + 0.5, y1=grid_shape[0] - 1 - r + 0.5,
                            line=dict(color='red', width=3)
                        ))
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=(self.model.layer_activation_gradients[connected_layer])[::-1, :],
            colorscale='Viridis',
            zmin=(self.model.layer_activation_gradients[connected_layer]).min(),
            zmax=(self.model.layer_activation_gradients[connected_layer]).max(),
            colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        ))
        try:
            xmin, xmax = ((region_settings[connected_layer])[0]) - 0.5, ((region_settings[connected_layer])[1]) + 0.5
            ymin, ymax = ((region_settings[connected_layer])[2]) - 0.5, ((region_settings[connected_layer])[3]) + 0.5
            heatmap_fig.update_layout(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax])
            )
        except Exception as e:
            pass

        def on_x_range_change(layout, x_range):
            output_xmin_widget.value = int(x_range[0])
            output_xmax_widget.value = int(x_range[1])
        def on_y_range_change(change, y_range):
            output_ymin_widget.value = int(y_range[0])
            output_ymax_widget.value = int(y_range[1])
        heatmap_fig.layout.on_change(on_x_range_change, 'xaxis.range')
        heatmap_fig.layout.on_change(on_y_range_change, 'yaxis.range')

        heatmap_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            width=300, height=300,
            xaxis=dict(dtick=1),
            yaxis=dict(dtick=1),
            shapes=highlight_shapes,
            annotations=[dict(
                text="Output-Layer Gradient",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10)
            )]
        )
        return heatmap_fig

    def _create_heatmap_target(self, current_node_index, current_layer, region_settings, offset_params):
        row_offset = -offset_params['target_row']
        col_offset = offset_params['target_col']
        shapes = []
        grid_shape = self.model.target_representation.shape
        if current_node_index is not None:
            row, col = divmod(current_node_index, self.model.layer_activation_gradients[current_layer].shape[-1])
            shapes.append(dict(
                type='circle', xref='x', yref='y',
                x0=col - col_offset - 0.5, y0=grid_shape[0]-1 - row - row_offset - 0.5,
                x1=col - col_offset + 0.5, y1=grid_shape[0]-1 - row - row_offset + 0.5,
                line=dict(color='red', width=3)
            ))
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=self.model.target_representation[::-1, :],
            colorscale='Viridis',
            zmin=self.model.target_representation.min(),
            zmax=self.model.target_representation.max(),
            colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        ))
        
        try:
            xmin, xmax = ((region_settings[current_layer])[0]) - col_offset - 0.5, ((region_settings[current_layer])[1]) - col_offset + 0.5
            ymin, ymax = ((grid_shape[0] - self.model.layer_activation_gradients[current_layer].shape[0]) + ((region_settings[current_layer])[2]) - row_offset - 0.5), ((grid_shape[0] - self.model.layer_activation_gradients[current_layer].shape[0]) + ((region_settings[current_layer])[3]) - row_offset + 0.5)
            heatmap_fig.update_layout(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax])
            )
        except Exception as e:
            pass
        
        heatmap_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            width=300, height=300,
            xaxis=dict(dtick=1),
            yaxis=dict(dtick=1),
            shapes=shapes,
            annotations=[dict(
                text="Target Representation",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10)
            )]
        )
        return heatmap_fig

    def _create_heatmap_input(self, current_node_index, current_layer, region_settings, offset_params):
        input_row_offset = -offset_params['input_row']
        input_col_offset = offset_params['input_col']
        shapes = []
        grid_shape = self.model.editable_input_representation.shape
        if current_node_index is not None:
            row, col = divmod(current_node_index, self.model.layer_activation_gradients[current_layer].shape[-1])
            shapes.append(dict(
                type='circle', xref='x', yref='y',
                x0=col - input_col_offset - 0.5, y0=grid_shape[0]-1 - row - input_row_offset - 0.5,
                x1=col - input_col_offset + 0.5, y1=grid_shape[0]-1 - row - input_row_offset + 0.5,
                line=dict(color='red', width=3)
            ))
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=self.model.editable_input_representation[::-1, :],
            colorscale=[[0, 'white'], [1, 'black']],
            zmin=0, zmax=1,
            colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        ))
        
        try:
            xmin, xmax = ((region_settings[current_layer])[0]) - input_col_offset - 0.5, ((region_settings[current_layer])[1]) - input_col_offset + 0.5
            ymin, ymax = ((grid_shape[0] - self.model.layer_activation_gradients[current_layer].shape[0]) + ((region_settings[current_layer])[2]) - input_row_offset - 0.5), ((grid_shape[0] - self.model.layer_activation_gradients[current_layer].shape[0]) + ((region_settings[current_layer])[3]) - input_row_offset + 0.5)
            heatmap_fig.update_layout(
                xaxis=dict(range=[xmin, xmax]),
                yaxis=dict(range=[ymin, ymax])
            )
        except Exception as e:
            pass
        
        heatmap_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            width=300, height=300,
            xaxis=dict(dtick=1),
            yaxis=dict(dtick=1),
            shapes=shapes,
            annotations=[dict(
                text="Input Representation",
                x=0.5 + input_col_offset/100.0, y=-0.15 + input_row_offset/100.0,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10)
            )]
        )
        return heatmap_fig

    def create_all_heatmaps(self, current_node_index, current_layer, connected_layer, threshold_value, offset_params, region_settings, 
                            select_xmin_widget, select_xmax_widget, select_ymin_widget, select_ymax_widget, 
                            output_xmin_widget, output_xmax_widget, output_ymin_widget, output_ymax_widget):
        conv1_heatmap = self._create_heatmap_conv1(current_node_index, current_layer, region_settings, select_xmin_widget, select_xmax_widget, select_ymin_widget, select_ymax_widget,)
        conv2_heatmap = self._create_heatmap_conv2(current_node_index, current_layer, connected_layer, threshold_value, region_settings, output_xmin_widget, output_xmax_widget, output_ymin_widget, output_ymax_widget)
        target_heatmap = self._create_heatmap_target(current_node_index, current_layer, region_settings, offset_params)
        input_heatmap = self._create_heatmap_input(current_node_index, current_layer, region_settings, offset_params)
        return conv1_heatmap, conv2_heatmap, target_heatmap, input_heatmap
