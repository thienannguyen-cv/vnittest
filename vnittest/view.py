# ======================================================
# View: Visualization creation (Sankey and Heatmaps)
# ======================================================
from vnittest import GradientModel
import numpy as np
import ipywidgets as widgets
import plotly.graph_objs as go
from IPython.display import display

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
    
    def create_sankey(self, current_layer, current_node_index, threshold_value, region_settings, sankey_sources, sankey_targets, sankey_values, sankey_container=None):
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

            for idx, src in enumerate(sankey_sources):
                if src in new_index_mapping and sankey_targets[idx] in new_index_mapping:
                    new_sources.append(new_index_mapping[src])
                    new_targets.append(new_index_mapping[sankey_targets[idx]])
                    new_values.append(sankey_values[idx])
                    if src == self.model.global_node_indices[(current_layer, self.model.layer_node_positions[current_layer][current_node_index])] and float(sankey_values[idx]) >= threshold_value:
                        new_link_colors.append("red")
                    else:
                        new_link_colors.append(f"rgba({128*int(float(sankey_values[idx])>threshold_value)},"
                                                 f"{128*int(float(sankey_values[idx])>threshold_value)},"
                                                 f"{128*int(float(sankey_values[idx])>threshold_value)},"
                                                 f"{1.*int(float(sankey_values[idx])>threshold_value)+.1})")
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
    
    def _calc_w(self, z, ids, current_zoom):
        from vnittest import utils
        
        id_z, row, col = 0, 0, 0
        if ids is not None:
            id_z, row, col = (ids[0]), (ids[1]), (ids[2])
        ch_id = (int(current_zoom*10)%10)
        w = (z[4:5,:,:,:])
        
        if z.shape[0] == 3*3:
            if ch_id!=9:
                w = utils.shift_tensor(z, ch_id)
            else:
                for i in range(3*3):
                    dx, dy = utils.INDEX2OFFSETS[i]
                    new_row, new_col = (row+dy), (col+dx)
                    if (new_row>0) and (new_row<(z.shape[-2])) and (new_col>0) and (new_col<(z.shape[-1])):
                        w[0,id_z,new_row,new_col] = (z[i,id_z,row,col])
            
        w = (w[0,id_z,:,:])
        return w
        
    def _create_heatmap_focused_conv(self, current_node_index, focused_layer, current_zoom, z, mask_ids, region_setting, xmin_widget, xmax_widget, ymin_widget, ymax_widget, is_conv1_choosed=True):
        from vnittest import utils
        
        ch_id = 0
        highlight_shapes = []
        grid_shape = self.model.layer_activation_gradients[focused_layer].shape
        xmin, xmax, ymin, ymax = (region_setting[0]), (region_setting[1]), (region_setting[2]), (region_setting[3])
        w = None
        if current_node_index is not None:
            ids = utils.calculate_conv_ids(mask_ids, current_node_index, grid_shape)
            id_z, row, col = (ids[0]), (ids[1]), (ids[2])
            
            w = 0.+z[4,id_z,:,:]
            ch_id = (int(current_zoom*10)%10)
            
            
            if ch_id != (3*3):
                w = 0.+z[ch_id,id_z,:,:]
                if utils.DEBUG_FLAG:
                    self.model.in_ch_id = ch_id
                    self.model.in_z_id = id_z
                    self.model.in_row = row
                    self.model.in_col = col
                    self.model.in_val = w[row, col]
            else:
                for i in range(3*3):
                    offsets = (utils.INDEX2OFFSETS[i])
                    w[row+(offsets[-2]), col+(offsets[-1])] = z[i,id_z,row, col]
                    highlight_shapes.append(dict(
                        type='circle', xref='x', yref='y',
                        x0=col + (offsets[-2]) - 0.5, y0=(z.shape[-2]) - 1 - row - (offsets[-1])  - 0.5,
                        x1=col + (offsets[-2])  + 0.5, y1=(z.shape[-2]) - 1 - row - (offsets[-1])  + 0.5,
                        line=dict(color='blue', width=3)
                    ))
            highlight_shapes.append(dict(
                type='circle', xref='x', yref='y',
                x0=col - 0.5, y0=(z.shape[-2]) - 1 - row - 0.5,
                x1=col + 0.5, y1=(z.shape[-2]) - 1 - row + 0.5,
                line=dict(color='red', width=3)
            ))
        colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=(w[::-1, :]),
            colorscale='Viridis',
            zmin=(w[::-1, :]).min(),
            zmax=(w[::-1, :]).max(),
            colorbar=colorbar, showscale=utils.is_show_color_bar
        ))
        try:
            heatmap_fig.update_layout(
                xaxis=dict(range=[xmin - 0.5, xmax + 0.5]),
                yaxis=dict(range=[ymin - 0.5, ymax + 0.5])
            )
        except Exception as e:
            pass

        def on_x_range_change(layout, x_range):
            h, w = ((mask_ids[-1]).shape[-2]), ((mask_ids[-1]).shape[-1])
            offsets = np.where(np.ones_like(mask_ids[-1])==1)
            mask = (((offsets[-1])>=((x_range[-2])-.5)) & ((offsets[-1])<=((x_range[-1])+.5))).reshape(-1,h,w)
            
            xmin, xmax = ((mask_ids[-1])[mask]).min(), ((mask_ids[-1])[mask]).max()
            xmin_widget.value = int(xmin)
            xmax_widget.value = int(xmax)
            
        def on_y_range_change(layout, y_range):
            h, w = ((mask_ids[-2]).shape[-2]), ((mask_ids[-2]).shape[-1])
            offsets = np.where(np.ones_like(mask_ids[-2])==1)
            mask = (((offsets[-2])>=((y_range[-2])-.5)) & ((offsets[-2])<=((y_range[-1])+.5))).reshape(-1,h,w)
            if (y_range[-2]) >= (y_range[-1]):
                print((y_range[-2], y_range[-1]))
                print(offsets[-2])
                print(offsets[-1])
                print(mask_ids[-2])
                print((mask_ids[-2])[mask])
            ymin, ymax = ((mask_ids[-2])[mask]).min(), ((mask_ids[-2])[mask]).max()
            ymin_widget.value = int(ymin)
            ymax_widget.value = int(ymax)
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
        if np.abs(xmax_widget.value-xmin_widget.value)>15:
            heatmap_fig.update_xaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(xmax_widget.value-xmin_widget.value))
            )
        if np.abs(ymax_widget.value-ymin_widget.value)>15:
            heatmap_fig.update_yaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(ymax_widget.value-ymin_widget.value))
            )
        return heatmap_fig

    def _create_heatmap_relative_conv(self, current_node_index, focused_layer, relative_layer, threshold_value, current_zoom, z, mask_ids, region_setting, xmin_widget, xmax_widget, ymin_widget, ymax_widget, is_conv1_choosed=True):
        from vnittest import utils
        
        ch_id = 0
        highlight_shapes = []
        grid_shape = None
        xmin, xmax, ymin, ymax = (region_setting[0]), (region_setting[1]), (region_setting[2]), (region_setting[3])
        ids, w = None, None
        current_layer, connected_layer = focused_layer, relative_layer
        if not is_conv1_choosed:
            current_layer, connected_layer = relative_layer, focused_layer
        if (self.model.flow_matrices[(current_layer, connected_layer)]) is not None:
            if not isinstance(self.model.flow_matrices[(current_layer, connected_layer)], dict):
                flow_matrix_thresholded = None
                if is_conv1_choosed:
                    grid_shape = self.model.layer_activation_gradients[connected_layer].shape
                    flow_matrix_thresholded = (((self.model.flow_matrices[(current_layer, connected_layer)]) > threshold_value)[current_node_index, :]).reshape(*grid_shape)
                else:
                    grid_shape = self.model.layer_activation_gradients[current_layer].shape
                    flow_matrix_thresholded = (((self.model.flow_matrices[(current_layer, connected_layer)]) > threshold_value)[:, current_node_index]).reshape(*grid_shape)
                for r in range(grid_shape[0]):
                    for c in range(grid_shape[1]):
                        if (flow_matrix_thresholded[r, c]) > 0:
                            highlight_shapes.append(dict(
                                type='circle', xref='x', yref='y',
                                x0=c - 0.5, y0=grid_shape[0] - 1 - r - 0.5,
                                x1=c + 0.5, y1=grid_shape[0] - 1 - r + 0.5,
                                line=dict(color='red', width=3)
                            ))
            else:
                flow_matrix = (self.model.flow_matrices[(current_layer, connected_layer)])
                grid_shape = self.model.layer_activation_gradients[relative_layer].shape
                if current_node_index is not None:
                    ids = utils.calculate_conv_ids(mask_ids, current_node_index, grid_shape)
                    id_z, row, col = (ids[0]), (ids[1]), (ids[2])
                    original_row, original_col = ((mask_ids[0])[id_z, row, col]), ((mask_ids[1])[id_z, row, col])
                    ch_id = (int(current_zoom*10)%10)
                    ch_ids = [ch_id]
                    w = 0.+z[4,id_z,:,:]
                    if ch_id==(3*3):
                        ch_ids = list(range(3*3))
                        ch_ids.append(4)
                    else:
                        w = 0.+z[ch_id,id_z,:,:]
                    
                    for j in ch_ids:
                        offsets = (utils.INDEX2OFFSETS[j])
                        new_row, new_col = (row+(offsets[-2])), (col+(offsets[-1]))
                        w[new_row, new_col] = z[j,id_z,new_row, new_col]
                        if utils.DEBUG_FLAG and (ch_id!=(3*3)):
                            self.model.out_ch_id = ch_id
                            self.model.out_id_z = id_z
                            self.model.out_row = new_row
                            self.model.out_col = new_col
                            self.model.out_val = w[new_row, new_col]
                        if (new_row>=0) and (new_row<((mask_ids[0]).shape[-2])) and (new_col>=0) and (new_col<((mask_ids[0]).shape[-1])):
                            color = ("blue" if j!=(ch_ids[-1]) else "red")
                            new_original_row, new_original_col = ((mask_ids[0])[id_z, new_row, new_col]), ((mask_ids[1])[id_z, new_row, new_col])
                            flow_value = 0
                            if is_conv1_choosed:
                                if ((original_row, original_col), (new_original_row, new_original_col)) in flow_matrix:
                                    flow_value = (flow_matrix[((original_row, original_col), (new_original_row, new_original_col))])
                            else:
                                if ((new_original_row, new_original_col), (original_row, original_col)) in flow_matrix:
                                    flow_value = (flow_matrix[((new_original_row, new_original_col), (original_row, original_col))])
                            if flow_value >= threshold_value:
                                highlight_shapes.append(dict(
                                    type='circle', xref='x', yref='y',
                                    x0=new_col - 0.5, y0=(z.shape[-2]) - 1 - new_row - 0.5,
                                    x1=new_col + 0.5, y1=(z.shape[-2]) - 1 - new_row + 0.5,
                                    line=dict(color=color, width=3)
                                ))
        colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=(w[::-1, :]),
            colorscale='Viridis',
            zmin=(w[::-1, :]).min(),
            zmax=(w[::-1, :]).max(),
            colorbar=colorbar, showscale=utils.is_show_color_bar
        ))
        try:
            heatmap_fig.update_layout(
                xaxis=dict(range=[xmin - 0.5, xmax + 0.5]),
                yaxis=dict(range=[ymin - 0.5, ymax + 0.5])
            )
        except Exception as e:
            pass

        def on_x_range_change(layout, x_range):
            h, w = ((mask_ids[-1]).shape[-2]), ((mask_ids[-1]).shape[-1])
            offsets = np.where(np.ones_like(mask_ids[-1])==1)
            mask = (((offsets[-1])>=((x_range[-2])-.5)) & ((offsets[-1])<=((x_range[-1])+.5))).reshape(-1,h,w)
            
            xmin, xmax = ((mask_ids[-1])[mask]).min(), ((mask_ids[-1])[mask]).max()
            xmin_widget.value = int(xmin)
            xmax_widget.value = int(xmax)
            
        def on_y_range_change(layout, y_range):
            h, w = ((mask_ids[-2]).shape[-2]), ((mask_ids[-2]).shape[-1])
            offsets = np.where(np.ones_like(mask_ids[-2])==1)
            mask = (((offsets[-2])>=((y_range[-2])-.5)) & ((offsets[-2])<=((y_range[-1])+.5))).reshape(-1,h,w)
            
            ymin, ymax = ((mask_ids[-2])[mask]).min(), ((mask_ids[-2])[mask]).max()
            ymin_widget.value = int(ymin)
            ymax_widget.value = int(ymax)
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
        if np.abs(xmax_widget.value-xmin_widget.value)>15:
            heatmap_fig.update_xaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(xmax_widget.value-xmin_widget.value))
            )
        if np.abs(ymax_widget.value-ymin_widget.value)>15:
            heatmap_fig.update_yaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(ymax_widget.value-ymin_widget.value))
            )
        return heatmap_fig

    def _create_heatmap_target(self, current_node_index, current_layer, current_zoom, z, mask_ids, region_setting):
        from vnittest import utils
        
        ch_id = 0
        highlight_shapes = []
        grid_shape = self.model.target_representation.shape
        xmin, xmax, ymin, ymax = (region_setting[0]), (region_setting[1]), (region_setting[2]), (region_setting[3])
        w = None
        if current_node_index is not None:
            ids = utils.calculate_conv_ids(mask_ids, current_node_index, grid_shape)
            id_z, row, col = (ids[0]), (ids[1]), (ids[2])
            original_row, original_col = ((mask_ids[0])[id_z, row, col]), ((mask_ids[1])[id_z, row, col])
            ch_id = (int(current_zoom*10)%10)
            w = np.max(z[0,:,:,:],axis=0,keepdims=False)
            if ch_id != (3*3):
                offsets = (utils.INDEX2OFFSETS[ch_id])
                new_row, new_col = (row+(offsets[-2])), (col+(offsets[-1]))
                if utils.DEBUG_FLAG:
                    self.model.target_ch_id = 0
                    self.model.target_id_z = id_z
                    self.model.target_row = new_row
                    self.model.target_col = new_col
                    self.model.target_val = w[new_row, new_col]
                if (new_row>=0) and (new_row<((mask_ids[0]).shape[-2])) and (new_col>=0) and (new_col<((mask_ids[0]).shape[-1])):
                    new_original_row, new_original_col = ((mask_ids[0])[id_z, new_row, new_col]), ((mask_ids[1])[id_z, new_row, new_col])
                    highlight_shapes.append(dict(
                        type='circle', xref='x', yref='y',
                        x0=new_col - 0.5, y0=(z.shape[-2]) - 1 - new_row - 0.5,
                        x1=new_col + 0.5, y1=(z.shape[-2]) - 1 - new_row + 0.5,
                        line=dict(color='red', width=3)
                    ))
            else:
                for i in range(3*3):
                    offsets = (utils.INDEX2OFFSETS[i])
                    new_row, new_col = (row+(offsets[-2])), (col+(offsets[-1]))
                    color = "blue"
                    if i==4:
                        color = "red"
                    if (new_row>=0) and (new_row<((mask_ids[0]).shape[-2])) and (new_col>=0) and (new_col<((mask_ids[0]).shape[-1])):
                        new_original_row, new_original_col = ((mask_ids[0])[id_z, new_row, new_col]), ((mask_ids[1])[id_z, new_row, new_col])
                        highlight_shapes.append(dict(
                            type='circle', xref='x', yref='y',
                            x0=new_col - 0.5, y0=(z.shape[-2]) - 1 - new_row - 0.5,
                            x1=new_col + 0.5, y1=(z.shape[-2]) - 1 - new_row + 0.5,
                            line=dict(color=color, width=3)
                        ))
                    
        colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=(w[::-1, :]),
            colorscale='Viridis',
            zmin=(w[::-1, :]).min(),
            zmax=(w[::-1, :]).max(),
            colorbar=colorbar, showscale=utils.is_show_color_bar
        ))
        try:
            heatmap_fig.update_layout(
                xaxis=dict(range=[xmin - 0.5, xmax + 0.5]),
                yaxis=dict(range=[ymin - 0.5, ymax + 0.5])
            )
        except Exception as e:
            pass

        heatmap_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            width=300, height=300,
            xaxis=dict(dtick=1),
            yaxis=dict(dtick=1),
            shapes=highlight_shapes,
            annotations=[dict(
                text="Target Representation",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10)
            )]
        )
        if np.abs(xmax-xmin)>15:
            heatmap_fig.update_xaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(xmax-xmin))
            )
        if np.abs(ymax-ymin)>15:
            heatmap_fig.update_yaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(ymax-ymin))
            )
        return heatmap_fig

    def _create_heatmap_input(self, current_node_index, current_layer, current_zoom, z, mask_ids, region_setting):
        from vnittest import utils
        
        ch_id = 0
        highlight_shapes = []
        grid_shape = self.model.input_representation.shape
        xmin, xmax, ymin, ymax = (region_setting[0]), (region_setting[1]), (region_setting[2]), (region_setting[3])
        w = None
        if current_node_index is not None:
            ch_id = (int(current_zoom*10)%10)
            ids = utils.calculate_conv_ids(mask_ids, current_node_index, grid_shape)
            id_z, row, col = (ids[0]), (ids[1]), (ids[2])
            w = 0.+(z[4,id_z,:,:])
            if ch_id != (3*3):
                offsets = (utils.INDEX2OFFSETS[ch_id])
                new_row, new_col = (row + (offsets[-2])), (col + (offsets[-1]))
                w = 0.+(z[ch_id,id_z,:,:])
                if utils.DEBUG_FLAG:
                    self.model.input_ch_id = ch_id
                    self.model.input_id_z = id_z
                    self.model.input_row = new_row
                    self.model.input_col = new_col
                    self.model.input_val = w[new_row, new_col]
                highlight_shapes.append(dict(
                    type='circle', xref='x', yref='y',
                    x0=new_col - 0.5, y0=(z.shape[-2]) - 1 - new_row - 0.5,
                    x1=new_col + 0.5, y1=(z.shape[-2]) - 1 - new_row + 0.5,
                    line=dict(color='red', width=3)
                ))
            else:
                for i in range(3*3):
                    color = "blue"
                    if i==4:
                        color = "red"
                    offsets = (utils.INDEX2OFFSETS[i])
                    new_row, new_col = (row+(offsets[-2])), (col+(offsets[-1]))
                    w[new_row, new_col] = z[i,id_z,new_row, new_col]

                    highlight_shapes.append(dict(
                        type='circle', xref='x', yref='y',
                        x0=col + (offsets[-1]) - 0.5, y0=(z.shape[-2]) - 1 - row - (offsets[-2]) - 0.5,
                        x1=col + (offsets[-1]) + 0.5, y1=(z.shape[-2]) - 1 - row - (offsets[-2]) + 0.5,
                        line=dict(color=color, width=3)
                    ))
            
        colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=(w[::-1, :]),
            colorscale='Viridis',
            zmin=(w[::-1, :]).min(),
            zmax=(w[::-1, :]).max(),
            colorbar=colorbar, showscale=utils.is_show_color_bar
        ))
        
        try:
            heatmap_fig.update_layout(
                xaxis=dict(range=[xmin - 0.5, xmax + 0.5]),
                yaxis=dict(range=[ymin - 0.5, ymax + 0.5])
            )
        except Exception as e:
            pass
        
        heatmap_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=40),
            width=300, height=300,
            xaxis=dict(dtick=1),
            yaxis=dict(dtick=1),
            shapes=highlight_shapes,
            annotations=[dict(
                text="Input Representation",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10)
            )]
        )
        if np.abs(xmax-xmin)>15:
            heatmap_fig.update_xaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(xmax-xmin))
            )
        if np.abs(ymax-ymin)>15:
            heatmap_fig.update_yaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(ymax-ymin))
            )
        return heatmap_fig
    
    def _create_heatmap_full_target(self, current_node_index, current_layer, current_zoom, z, mask_ids, region_setting):
        from vnittest import utils
        
        shapes = []
        grid_shape = self.model.target_representation.shape
        if current_node_index is not None:
            ids = utils.calculate_conv_ids(mask_ids, current_node_index, grid_shape)
            id_z, row, col = (ids[0]), (ids[1]), (ids[2])
            original_row, original_col = ((mask_ids[0])[id_z, row, col]), ((mask_ids[1])[id_z, row, col])
            ch_id = (int(current_zoom*10)%10)
            if ch_id==(3*3):
                ch_id = 4
            offsets = (utils.INDEX2OFFSETS[ch_id])
            new_row, new_col = (row+(offsets[-2])), (col+(offsets[-1]))
            if (new_row>=0) and (new_row<((mask_ids[0]).shape[-2])) and (new_col>=0) and (new_col<((mask_ids[0]).shape[-1])):
                new_original_row, new_original_col = ((mask_ids[0])[id_z, new_row, new_col]), ((mask_ids[1])[id_z, new_row, new_col])
                
                shapes.append(dict(
                    type='circle', xref='x', yref='y',
                    x0=new_original_col - 0.5, y0=grid_shape[-2]-1 - new_original_row - 0.5,
                    x1=new_original_col + 0.5, y1=grid_shape[-2]-1 - new_original_row + 0.5,
                    line=dict(color='red', width=3)
                ))
        w = 0.+self.model.target_representation[0,:,:]
        colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=w[::-1, :],
            colorscale='Viridis',
            zmin=(w[::-1, :]).min(),
            zmax=(w[::-1, :]).max(),
            colorbar=colorbar, showscale=utils.is_show_color_bar
        ))
        
        try:
            xmin, xmax = (region_setting[0]) - 0.5, (region_setting[1]) + 0.5
            ymin, ymax = ((grid_shape[-2] - (self.model.layer_activation_gradients[current_layer]).shape[-2]) + (region_setting[2]) - 0.5), ((grid_shape[-2] - (self.model.layer_activation_gradients[current_layer]).shape[-2]) + (region_setting[3]) + 0.5)
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
                text="Full-sized Target Representation",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10)
            )]
        )
        if np.abs(xmax-xmin)>15:
            heatmap_fig.update_xaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(xmax-xmin))
            )
        if np.abs(ymax-ymin)>15:
            heatmap_fig.update_yaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(ymax-ymin))
            )
        return heatmap_fig

    def _create_heatmap_full_input(self, current_node_index, current_layer, current_zoom, z, mask_ids, region_setting):
        from vnittest import utils
        
        ch_id = 0
        shapes = []
        grid_shape = self.model.editable_input_representation.shape
        if current_node_index is not None:
            ids = utils.calculate_conv_ids(mask_ids, current_node_index, grid_shape)
            id_z, row, col = (ids[0]), (ids[1]), (ids[2])
            original_row, original_col = ((mask_ids[0])[id_z, row, col]), ((mask_ids[1])[id_z, row, col])
            ch_id = (int(current_zoom*10)%10)
            if ch_id==(3*3):
                ch_id = 4
            offsets = (utils.INDEX2OFFSETS[ch_id])
            new_row, new_col = (row+(offsets[-2])), (col+(offsets[-1]))
            if (new_row>=0) and (new_row<((mask_ids[0]).shape[-2])) and (new_col>=0) and (new_col<((mask_ids[0]).shape[-1])):
                new_original_row, new_original_col = ((mask_ids[0])[id_z, new_row, new_col]), ((mask_ids[1])[id_z, new_row, new_col])
                
                shapes.append(dict(
                    type='circle', xref='x', yref='y',
                    x0=new_original_col - 0.5, y0=grid_shape[-2]-1 - new_original_row - 0.5,
                    x1=new_original_col + 0.5, y1=grid_shape[-2]-1 - new_original_row + 0.5,
                    line=dict(color='red', width=3)
                ))
        temp_shape = self.model.editable_input_representation.shape
        w = 0.+self.model.editable_input_representation.reshape(temp_shape[0],1,temp_shape[-2],temp_shape[-1])[ch_id,0,:,:]
        colorbar=dict(title='', len=0.5, x=1.02, xanchor='left', thickness=10)
        heatmap_fig = go.FigureWidget(data=go.Heatmap(
            z=np.abs(w[::-1, :]),
            colorscale=[[0, 'white'], [1, 'black']],
            zmin=0, zmax=1,
            colorbar=colorbar, showscale=utils.is_show_color_bar
        ))
        
        try:
            xmin, xmax = (region_setting[0]) - 0.5, (region_setting[1]) + 0.5
            ymin, ymax = ((grid_shape[-2] - self.model.layer_activation_gradients[current_layer].shape[-2]) + (region_setting[2]) - 0.5), ((grid_shape[-2] - self.model.layer_activation_gradients[current_layer].shape[-2]) + (region_setting[3]) + 0.5)
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
                text="Full-sized Input Representation",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=10)
            )]
        )
        if np.abs(xmax-xmin)>15:
            heatmap_fig.update_xaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(xmax-xmin))
            )
        if np.abs(ymax-ymin)>15:
            heatmap_fig.update_yaxes(
                tickmode='array',
                nticks=10
            )
        else:
            heatmap_fig.update_xaxes(
                dtick=1,
                nticks=int(np.abs(ymax-ymin))
            )
        return heatmap_fig

    def create_all_heatmaps(self, current_node_index, current_zoom, selected_layer, connected_layer, threshold_value, conv_zs, conv_target, conv_input, conv_mask_ids, conv_region_settings, region_settings, offset_params, select_xmin_widget, select_xmax_widget, select_ymin_widget, select_ymax_widget, output_xmin_widget, output_xmax_widget, output_ymin_widget, output_ymax_widget, is_conv1_choosed=True):
        conv1_heatmap, conv2_heatmap = None, None
        if is_conv1_choosed:
            row_offset, col_offset = offset_params[selected_layer]
            conv1_heatmap = self._create_heatmap_focused_conv(current_node_index+row_offset*(self.model.layer_activation_gradients[selected_layer].shape[-1])+col_offset, selected_layer, current_zoom, conv_zs[selected_layer], conv_mask_ids[selected_layer], conv_region_settings[selected_layer], select_xmin_widget, select_xmax_widget, select_ymin_widget, select_ymax_widget, is_conv1_choosed=is_conv1_choosed)
            
            row_offset, col_offset = offset_params[connected_layer]
            conv2_heatmap = self._create_heatmap_relative_conv(current_node_index+row_offset*(self.model.layer_activation_gradients[connected_layer].shape[-1])+col_offset, selected_layer, connected_layer, threshold_value, current_zoom, conv_zs[connected_layer], conv_mask_ids[connected_layer], conv_region_settings[connected_layer], output_xmin_widget, output_xmax_widget, output_ymin_widget, output_ymax_widget, is_conv1_choosed=is_conv1_choosed)
        else:
            row_offset, col_offset = offset_params[selected_layer]
            conv1_heatmap = self._create_heatmap_relative_conv(current_node_index+row_offset*(self.model.layer_activation_gradients[selected_layer].shape[-1])+col_offset, connected_layer, selected_layer, threshold_value, current_zoom, conv_zs[selected_layer], conv_mask_ids[selected_layer], conv_region_settings[selected_layer], select_xmin_widget, select_xmax_widget, select_ymin_widget, select_ymax_widget, is_conv1_choosed=is_conv1_choosed)
            
            row_offset, col_offset = offset_params[connected_layer]
            conv2_heatmap = self._create_heatmap_focused_conv(current_node_index+row_offset*(self.model.layer_activation_gradients[connected_layer].shape[-1])+col_offset, connected_layer, current_zoom, conv_zs[connected_layer], conv_mask_ids[connected_layer], conv_region_settings[connected_layer], output_xmin_widget, output_xmax_widget, output_ymin_widget, output_ymax_widget, is_conv1_choosed=is_conv1_choosed)
            
        row_offset, col_offset = offset_params[connected_layer]
        target_heatmap = self._create_heatmap_target(current_node_index+row_offset*(self.model.target_representation.shape[-1])+col_offset, connected_layer, current_zoom, conv_target, conv_mask_ids[connected_layer], conv_region_settings[connected_layer])
        
        row_offset, col_offset = offset_params[selected_layer]
        input_heatmap = self._create_heatmap_input(current_node_index+row_offset*(self.model.input_representation.shape[-1])+col_offset, selected_layer, current_zoom, conv_input, conv_mask_ids[selected_layer], conv_region_settings[selected_layer])
        
        row_offset, col_offset = offset_params[connected_layer]
        full_target_heatmap = self._create_heatmap_full_target(current_node_index+row_offset*(self.model.target_representation.shape[-1])+col_offset, selected_layer, current_zoom, conv_target, conv_mask_ids[connected_layer], region_settings[connected_layer])
        
        row_offset, col_offset = offset_params[selected_layer]
        full_input_heatmap = self._create_heatmap_full_input(current_node_index+row_offset*(self.model.editable_input_representation.shape[-1])+col_offset, selected_layer, current_zoom, conv_input, conv_mask_ids[selected_layer], region_settings[selected_layer])
        return conv1_heatmap, conv2_heatmap, target_heatmap, input_heatmap, full_target_heatmap, full_input_heatmap
