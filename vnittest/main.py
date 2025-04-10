from model import GradientModel
from view import Visualizer
from controller import UIController

# ======================================================
# Facade: Tool wrapper that packages Model, Visualizer and UIController
# ======================================================
class GradientFlowInteractive:
    def __init__(self, debug_folder):
        self.model = GradientModel(debug_folder, self.progress_bar)
        self.visualizer = Visualizer(self.model)
        self.ui_controller = UIController(self.model, self.visualizer)
        self.progress_bar = self.visualizer.get_progress_bar()
        
    def allocate_resource(self):
        self.progress_bar = visualizer.get_progress_bar()
        self.progress_bar.set_progress(0)
        visualizer.display_progress_bar()
        model.set_progress_bar(self.progress_bar)
        visualizer.allocate_resource()
        model.allocate_resource()

    def render_ipywidgets(self):
        """Render the tool using ipywidgets (for Jupyter Notebook)."""
        self.allocate_resource()
        self.progress_bar.set_progress(100)
        self.visualizer.destroy_progress_bar()
        self.ui_controller.render_ui()
        
    def render_webview(self, port=8050):
        pass

# ======================================================
# Demo Usage:
# ======================================================
if __name__ == '__main__':
    # Khởi tạo tool với folder debug (điều chỉnh đường dẫn nếu cần)
    tool = GradientFlowInteractive("./tests/test_data/test_")
    # Để render trong Jupyter, gọi:
    # tool.render_ipywidgets()

    # Để render trong WebView (ứng dụng Dash), gọi:
    tool.render_webview(port=8050)