import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from scipy.signal import wiener
from scipy.stats import entropy
from matplotlib.widgets import RectangleSelector

class ImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Astronomical Image Analyzer")
        
        # Set window icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'app_icon.png')
            if os.path.exists(icon_path):
                icon_image = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, icon_image)
            else:
                print("Icon file not found:", icon_path)
        except Exception as e:
            print("Could not load icon:", str(e))
        
        self.root.state('zoomed')
        
        self.image = None
        self.current_cmap = "gray"
        self.circularity_threshold = 0.7  
        self.roi_rect = None  
        self.setup_tabs()
        
    def setup_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both")
        
        self.single_image_tab = ttk.Frame(self.notebook)
        self.batch_processing_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)  
        
        self.notebook.add(self.single_image_tab, text="Single Image Processing")
        self.notebook.add(self.batch_processing_tab, text="Batch Processing")
        self.notebook.add(self.visualization_tab, text="Visualization")
        
        self.setup_single_image_tab()
        self.setup_batch_processing_tab()
        self.setup_visualization_tab()
        
    def setup_single_image_tab(self):
        main_container = ttk.Frame(self.single_image_tab, padding="10")
        main_container.pack(fill="both", expand=True)
        
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(fill="x", pady=(0, 10))
        
        left_controls = ttk.Frame(controls_frame)
        left_controls.pack(side="left", fill="x", expand=True)
        
        ttk.Button(left_controls, text="Load Image", command=self.load_image).pack(side="left", padx=5)
        ttk.Button(left_controls, text="Original Image", command=lambda: self.process_image(1)).pack(side="left", padx=5)
        ttk.Button(left_controls, text="Object Contours", command=lambda: self.process_image(6)).pack(side="left", padx=5)
        
        circularity_frame = ttk.Frame(left_controls)
        circularity_frame.pack(side="left", padx=5)
        ttk.Label(circularity_frame, text="Circularity Threshold:").pack(side="left")
        self.threshold_var = tk.StringVar(value="0.7")
        threshold_entry = ttk.Entry(circularity_frame, textvariable=self.threshold_var, width=5)
        threshold_entry.pack(side="left", padx=2)
        ttk.Button(circularity_frame, text="Circularity", command=self.update_circularity_threshold).pack(side="left")
        
        right_controls = ttk.Frame(controls_frame)
        right_controls.pack(side="right", fill="x", expand=True)
        
        ttk.Label(right_controls, text="Colormap:").pack(side="left", padx=5)
        self.cmap_var = tk.StringVar(value="gray")
        cmap_combo = ttk.Combobox(right_controls, textvariable=self.cmap_var, 
                                values=["gray", "viridis", "plasma", "inferno", "magma", "hot"],
                                state="readonly", width=10)
        cmap_combo.pack(side="left", padx=5)
        cmap_combo.bind("<<ComboboxSelected>>", self.update_colormap)
        
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill="both", expand=True)
        
        display_frame = ttk.Frame(content_frame)
        display_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, display_frame)
        self.toolbar.pack(fill="x")
        
        info_frame = ttk.Frame(content_frame, relief="solid", borderwidth=1)
        info_frame.pack(side="right", fill="y", padx=(0, 5))
        
        ttk.Label(info_frame, text="Image Information", font=("Arial", 10, "bold")).pack(pady=(5, 5))
        
        dims_frame = ttk.Frame(info_frame, relief="solid", borderwidth=1)
        dims_frame.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(dims_frame, text="Dimensions:", font=("Arial", 9, "bold")).pack(anchor="w", padx=5)
        self.dims_label = ttk.Label(dims_frame, text="No image loaded", padding=5)
        self.dims_label.pack(fill="x")
        
        size_frame = ttk.Frame(info_frame, relief="solid", borderwidth=1)
        size_frame.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(size_frame, text="File Size:", font=("Arial", 9, "bold")).pack(anchor="w", padx=5)
        self.size_label = ttk.Label(size_frame, text="No image loaded", padding=5)
        self.size_label.pack(fill="x")
        
        intensity_frame = ttk.Frame(info_frame, relief="solid", borderwidth=1)
        intensity_frame.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(intensity_frame, text="Intensity Statistics:", font=("Arial", 9, "bold")).pack(anchor="w", padx=5)
        self.intensity_label = ttk.Label(intensity_frame, text="No image loaded", padding=5)
        self.intensity_label.pack(fill="x")
        
        contour_frame = ttk.Frame(info_frame, relief="solid", borderwidth=1)
        contour_frame.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(contour_frame, text="Contour Statistics:", font=("Arial", 9, "bold")).pack(anchor="w", padx=5)
        self.contour_label = ttk.Label(contour_frame, text="No image loaded", padding=5)
        self.contour_label.pack(fill="x")
        
        circularity_frame = ttk.Frame(info_frame, relief="solid", borderwidth=1)
        circularity_frame.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(circularity_frame, text="Circularity Statistics:", font=("Arial", 9, "bold")).pack(anchor="w", padx=5)
        self.circularity_label = ttk.Label(circularity_frame, text="No image loaded", padding=5)
        self.circularity_label.pack(fill="x")
        
        coords_frame = ttk.Frame(info_frame, relief="solid", borderwidth=1)
        coords_frame.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(coords_frame, text="Cursor Position:", font=("Arial", 9, "bold")).pack(anchor="w", padx=5)
        self.coords_label = ttk.Label(coords_frame, text="(0, 0)", padding=5)
        self.coords_label.pack(fill="x")
        
        edge_density_frame = ttk.Frame(info_frame, relief="solid", borderwidth=1)
        edge_density_frame.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(edge_density_frame, text="Edge Analysis:", font=("Arial", 9, "bold")).pack(anchor="w", padx=5)
        self.edge_density_label = ttk.Label(edge_density_frame, text="No image loaded", padding=5)
        self.edge_density_label.pack(fill="x")
        
        entropy_frame = ttk.Frame(info_frame, relief="solid", borderwidth=1)
        entropy_frame.pack(fill="x", padx=5, pady=(0, 5))
        ttk.Label(entropy_frame, text="Image Complexity:", font=("Arial", 9, "bold")).pack(anchor="w", padx=5)
        self.entropy_label = ttk.Label(entropy_frame, text="No image loaded", padding=5)
        self.entropy_label.pack(fill="x")
        
        self.canvas.mpl_connect('motion_notify_event', self.update_cursor_coords)
        
    def update_circularity_threshold(self):
        try:
            new_threshold = float(self.threshold_var.get())
            if 0 <= new_threshold <= 1:
                self.circularity_threshold = new_threshold
                self.process_image(7)
            else:
                messagebox.showerror("Error", "Threshold must be between 0 and 1")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
        
    def calculate_circularity(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity
        
    def update_cursor_coords(self, event):
        if event.inaxes:
            self.coords_label.config(text=f"({int(event.xdata)}, {int(event.ydata)}) pixels")
        else:
            self.coords_label.config(text="(0, 0) pixels")
        
    def update_colormap(self, event=None):
        self.current_cmap = self.cmap_var.get()
        if self.image is not None:
            self.process_image(1)  # Refresh the display with new colormap
    
    def update_image_info(self):
        if self.image is not None:
            height, width = self.image.shape
            file_size = os.path.getsize(self.current_file) / (1024 * 1024)  # Convert to MB
            mean_intensity = np.mean(self.image)
            std_intensity = np.std(self.image)
            min_intensity = np.min(self.image)
            max_intensity = np.max(self.image)
            contours = self.detect_contours(self.image)
            
            largest_area = 0
            circularities = []
            if contours:
                largest_area = max(cv2.contourArea(contour) for contour in contours)
                circularities = [self.calculate_circularity(contour) for contour in contours]
            
            edge_density = self.calculate_edge_density(self.image)
            histogram_entropy = self.calculate_histogram_entropy(self.image)
            
            self.dims_label.config(text=f"{width}x{height} pixels")
            self.size_label.config(text=f"{file_size:.2f} MB")
            self.intensity_label.config(text=f"Mean: {mean_intensity:.2f} ADU\nStd Dev: {std_intensity:.2f} ADU\nMin: {min_intensity:.2f} ADU\nMax: {max_intensity:.2f} ADU")
            self.contour_label.config(text=f"Count: {len(contours)}\nLargest Area: {largest_area:.2f} pixels²")
            
            if circularities:
                avg_circularity = np.mean(circularities)
                max_circularity = max(circularities)
                min_circularity = min(circularities)
                circular_objects = sum(1 for c in circularities if c >= self.circularity_threshold)
                self.circularity_label.config(text=f"Average: {avg_circularity:.3f}\nMax: {max_circularity:.3f}\nMin: {min_circularity:.3f}\nCircular Objects: {circular_objects}")
            else:
                self.circularity_label.config(text="No objects detected")
                
            self.edge_density_label.config(text=f"Edge Density: {edge_density:.2f} ADU/pixel")
            self.entropy_label.config(text=f"Histogram Entropy: {histogram_entropy:.2f} bits")
        else:
            self.dims_label.config(text="No image loaded")
            self.size_label.config(text="No image loaded")
            self.intensity_label.config(text="No image loaded")
            self.contour_label.config(text="No image loaded")
            self.circularity_label.config(text="No image loaded")
            self.edge_density_label.config(text="No image loaded")
            self.entropy_label.config(text="No image loaded")
    
    def calculate_edge_density(self, image):
        img_float = image.astype(float)
        sobelx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.mean(gradient_magnitude)
        
        return edge_density
        
    def calculate_histogram_entropy(self, image):
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        
        hist = hist / hist.sum()
        entropy_value = entropy(hist)
        
        return entropy_value
    
    def setup_batch_processing_tab(self):
        batch_frame = ttk.Frame(self.batch_processing_tab, padding="10")
        batch_frame.pack(fill="both", expand=True)
        
        threshold_frame = ttk.Frame(batch_frame)
        threshold_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(threshold_frame, text="Circularity Threshold:").pack(side="left", padx=5)
        self.batch_threshold_var = tk.StringVar(value="0.7")
        ttk.Entry(threshold_frame, textvariable=self.batch_threshold_var, width=5).pack(side="left", padx=5)
        
        ttk.Button(batch_frame, text="Select Folder", command=self.select_folder).pack(pady=5)
        
        self.results_tree = ttk.Treeview(batch_frame, 
                                       columns=("Image Name", "Width", "Height", "Size (MB)", 
                                               "Mean Intensity", "Std Dev", "Contour Count", 
                                               "Largest Area", "Circular Objects", "Edge Density", 
                                               "Histogram Entropy"), 
                                       show="headings")
        # headings
        self.results_tree.heading("Image Name", text="Image Name")
        self.results_tree.heading("Width", text="Width (pixels)")
        self.results_tree.heading("Height", text="Height (pixels)")
        self.results_tree.heading("Size (MB)", text="Size (MB)")
        self.results_tree.heading("Mean Intensity", text="Mean Intensity (ADU)")
        self.results_tree.heading("Std Dev", text="Std Dev (ADU)")
        self.results_tree.heading("Contour Count", text="Contour Count")
        self.results_tree.heading("Largest Area", text="Largest Area (pixels²)")
        self.results_tree.heading("Circular Objects", text="Circular Objects")
        self.results_tree.heading("Edge Density", text="Edge Density (ADU/pixel)")
        self.results_tree.heading("Histogram Entropy", text="Histogram Entropy (bits)")
        
        self.results_tree.column("Image Name", width=200)
        self.results_tree.column("Width", width=100)
        self.results_tree.column("Height", width=100)
        self.results_tree.column("Size (MB)", width=100)
        self.results_tree.column("Mean Intensity", width=120)
        self.results_tree.column("Std Dev", width=100)
        self.results_tree.column("Contour Count", width=100)
        self.results_tree.column("Largest Area", width=120)
        self.results_tree.column("Circular Objects", width=100)
        self.results_tree.column("Edge Density", width=120)
        self.results_tree.column("Histogram Entropy", width=120)
        
        self.results_tree.pack(fill="both", expand=True)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_file = file_path
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.image is not None:
                self.process_image(1)  # Show the original image
                self.update_image_info()
                self.show_edge_detection()
                self.show_histogram()
                messagebox.showinfo("Success", "Image loaded successfully!")
            else:
                messagebox.showerror("Error", "Could not load the image.")
    
    def detect_contours(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def process_image(self, choice):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        self.ax.clear()
        
        if choice == 1:
            self.ax.imshow(self.image, cmap=self.current_cmap, zorder=1)
            self.ax.set_title("Original Astronomical Image")
            
            if hasattr(self, 'roi_coords') and self.roi_coords:
                x1, y1, x2, y2 = self.roi_coords
                if hasattr(self, 'roi_rect') and self.roi_rect is not None:
                    try:
                        self.roi_rect.remove()
                    except:
                        pass
                self.roi_rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                            fill=True,
                                            facecolor='yellow',
                                            edgecolor='red',
                                            alpha=0.3,
                                            linewidth=2,
                                            zorder=2)  
                self.ax.add_patch(self.roi_rect)
        
        elif choice == 6:
            contours = self.detect_contours(self.image)
            image_with_contours = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)
            self.ax.imshow(image_with_contours)
            self.ax.set_title(f"Detected Objects: {len(contours)}")
            
            if hasattr(self, 'roi_coords') and self.roi_coords:
                x1, y1, x2, y2 = self.roi_coords
                if hasattr(self, 'roi_rect') and self.roi_rect is not None:
                    try:
                        self.roi_rect.remove()
                    except:
                        pass
                self.roi_rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                            fill=True,
                                            facecolor='yellow',
                                            edgecolor='red',
                                            alpha=0.3,
                                            linewidth=2,
                                            zorder=2)
                self.ax.add_patch(self.roi_rect)
        
        elif choice == 7:
            contours = self.detect_contours(self.image)
            image_with_circularity = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            
            circular_count = 0
            for contour in contours:
                circularity = self.calculate_circularity(contour)
                if circularity >= self.circularity_threshold:
                    circular_count += 1
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                
                cv2.drawContours(image_with_circularity, [contour], -1, color, 1)
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(image_with_circularity, f"{circularity:.2f}", 
                              (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            self.ax.imshow(image_with_circularity)
            self.ax.set_title(f"Circularity Analysis (Threshold: {self.circularity_threshold:.2f})\nCircular Objects: {circular_count}")
            
            if hasattr(self, 'roi_coords') and self.roi_coords:
                x1, y1, x2, y2 = self.roi_coords
                if hasattr(self, 'roi_rect') and self.roi_rect is not None:
                    try:
                        self.roi_rect.remove()
                    except:
                        pass
                self.roi_rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                            fill=True,
                                            facecolor='yellow',
                                            edgecolor='red',
                                            alpha=0.3,
                                            linewidth=2,
                                            zorder=2)
                self.ax.add_patch(self.roi_rect)
        
        self.ax.axis("off")
        self.canvas.draw()
        
    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        
        try:
            threshold = float(self.batch_threshold_var.get())
            if not 0 <= threshold <= 1:
                messagebox.showerror("Error", "Threshold must be between 0 and 1")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid threshold value")
            return
        
        results = []
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    height, width = image.shape
                    file_size = os.path.getsize(image_path) / (1024 * 1024)  # Convert to MB
                    mean_intensity = np.mean(image)
                    std_intensity = np.std(image)
                    contours = self.detect_contours(image)
                    
                    largest_area = 0
                    circular_count = 0
                    if contours:
                        largest_area = max(cv2.contourArea(contour) for contour in contours)
                        circularities = [self.calculate_circularity(contour) for contour in contours]
                        circular_count = sum(1 for c in circularities if c >= threshold)
                    
                    edge_density = self.calculate_edge_density(image)
                    histogram_entropy = self.calculate_histogram_entropy(image)
                    
                    results.append((filename, width, height, f"{file_size:.2f}", 
                                  f"{mean_intensity:.2f}", f"{std_intensity:.2f}", 
                                  len(contours), f"{largest_area:.2f}", circular_count,
                                  f"{edge_density:.2f}", f"{histogram_entropy:.2f}"))
        
        self.update_results_table(results)
        pd.DataFrame(results, columns=["Image Name", "Width (pixels)", "Height (pixels)", "Size (MB)", 
                                     "Mean Intensity (ADU)", "Std Dev (ADU)", "Contour Count", 
                                     "Largest Area (pixels²)", "Circular Objects", 
                                     "Edge Density (ADU/pixel)", "Histogram Entropy (bits)"]).to_csv("batch_results.csv", index=False)
        messagebox.showinfo("Batch Processing Complete", "Results saved to batch_results.csv")
        
    def update_results_table(self, results):
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)
        
        for item in results:
            self.results_tree.insert("", "end", values=item)

    def setup_visualization_tab(self):
        main_container = ttk.Frame(self.visualization_tab, padding="10")
        main_container.pack(fill="both", expand=True)
        
        viz_notebook = ttk.Notebook(main_container)
        viz_notebook.pack(fill="both", expand=True)
    
        histogram_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(histogram_tab, text="Histogram")
        edge_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(edge_tab, text="Edge Detection")
        self.setup_histogram_tab(histogram_tab)
        self.setup_edge_detection_tab(edge_tab)
        
        self.roi_selector = None
        self.roi_coords = None
        
    def setup_histogram_tab(self, parent):
        controls_frame = ttk.LabelFrame(parent, text="Histogram Controls", padding="5")
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        display_frame = ttk.Frame(controls_frame)
        display_frame.pack(fill="x", padx=5, pady=5)
        
        scale_frame = ttk.Frame(display_frame)
        scale_frame.pack(side="left", padx=10)
        
        self.log_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scale_frame, text="Log Scale", 
                       variable=self.log_scale_var,
                       command=self.update_histogram).pack(side="left")
        
        stats_frame = ttk.Frame(display_frame)
        stats_frame.pack(side="left", padx=10)
        
        self.show_mean_var = tk.BooleanVar(value=True)
        self.show_median_var = tk.BooleanVar(value=True)
        self.show_std_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(stats_frame, text="Show Mean", 
                       variable=self.show_mean_var,
                       command=self.update_histogram).pack(side="left")
        ttk.Checkbutton(stats_frame, text="Show Median", 
                       variable=self.show_median_var,
                       command=self.update_histogram).pack(side="left")
        ttk.Checkbutton(stats_frame, text="Show Std Dev", 
                       variable=self.show_std_var,
                       command=self.update_histogram).pack(side="left")
        
        save_frame = ttk.Frame(display_frame)
        save_frame.pack(side="right", padx=10)
        ttk.Button(save_frame, text="Save Plot", 
                  command=self.save_plot).pack(side="right")
        
        roi_frame = ttk.Frame(display_frame)
        roi_frame.pack(side="right", padx=10)
        ttk.Button(roi_frame, text="Select ROI", 
                  command=self.toggle_roi_selector).pack(side="right")
        ttk.Button(roi_frame, text="Reset ROI", 
                  command=self.reset_roi).pack(side="right", padx=5)
        
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(10, 6))
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=parent)
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        self.hist_toolbar = NavigationToolbar2Tk(self.hist_canvas, parent)
        self.hist_toolbar.pack(fill="x")
        
        self.cursor_info = ttk.Label(parent, text="")
        self.cursor_info.pack(fill="x", padx=5)
        
        self.hist_canvas.mpl_connect('motion_notify_event', self.on_histogram_hover)
        
    def setup_edge_detection_tab(self, parent):    
        controls_frame = ttk.LabelFrame(parent, text="Edge Detection Controls", padding="5")
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        threshold_frame = ttk.Frame(controls_frame)
        threshold_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(threshold_frame, text="Lower Threshold:").pack(side="left", padx=5)
        self.edge_low_threshold_var = tk.StringVar(value="100")
        ttk.Entry(threshold_frame, textvariable=self.edge_low_threshold_var, 
                 width=5).pack(side="left", padx=5)
        
        ttk.Label(threshold_frame, text="Upper Threshold:").pack(side="left", padx=5)
        self.edge_high_threshold_var = tk.StringVar(value="200")
        ttk.Entry(threshold_frame, textvariable=self.edge_high_threshold_var, 
                 width=5).pack(side="left", padx=5)
        
        ttk.Button(threshold_frame, text="Update Edge Detection", 
                  command=self.update_edge_detection).pack(side="left", padx=20)
        
        self.edge_fig, self.edge_ax = plt.subplots(figsize=(10, 6))
        self.edge_canvas = FigureCanvasTkAgg(self.edge_fig, master=parent)
        self.edge_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        self.edge_toolbar = NavigationToolbar2Tk(self.edge_canvas, parent)
        self.edge_toolbar.pack(fill="x")

        self.hist_toolbar._orig_home = self.hist_toolbar.home
        self.edge_toolbar._orig_home = self.edge_toolbar.home

        def new_hist_home(self):
            self._orig_home()
            if hasattr(self, 'hist_ax'):
                self.hist_ax.set_xlim(-1, 256)
                if self.log_scale_var.get():
                    self.hist_ax.set_yscale('log')
                    self.hist_ax.set_ylim(1, None)
                else:
                    self.hist_ax.set_yscale('linear')
                    self.hist_ax.set_ylim(0, None)
                self.hist_canvas.draw()
                if hasattr(self, 'roi_coords') and self.roi_coords:
                    self.update_histogram()

        def new_edge_home(self):
            self._orig_home()
            if self.image is not None:
                height, width = self.image.shape
                self.edge_ax.set_xlim(-1, width)
                self.edge_ax.set_ylim(height, -1)  
                self.edge_canvas.draw()
                if hasattr(self, 'roi_coords') and self.roi_coords:
                    self.update_edge_detection()

        self.hist_toolbar.home = new_hist_home.__get__(self.hist_toolbar)
        self.edge_toolbar.home = new_edge_home.__get__(self.edge_toolbar)

        def sync_views(event):
            if not hasattr(self, 'image') or self.image is None:
                return
                
            if event.inaxes == self.hist_ax:
                self.edge_ax.set_xlim(self.hist_ax.get_xlim())
                self.edge_canvas.draw()
            elif event.inaxes == self.edge_ax:
                self.hist_ax.set_xlim(self.edge_ax.get_xlim())
                self.hist_canvas.draw()

        self.hist_canvas.mpl_connect('motion_notify_event', sync_views)
        self.edge_canvas.mpl_connect('motion_notify_event', sync_views)

    def on_histogram_hover(self, event):
        if event.inaxes == self.hist_ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.cursor_info.config(
                    text=f"Intensity: {int(x)}, Count: {int(y):,}"
                )
        else:
            self.cursor_info.config(text="")

    def toggle_roi_selector(self):
        if self.roi_selector is None:
            self.roi_selector = RectangleSelector(
                self.ax,
                self.on_roi_select,
                interactive=True,
                props=dict(
                    facecolor='yellow',
                    edgecolor='red',
                    alpha=0.3,
                    fill=True,
                    linewidth=2
                ),
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                button=[1],
                useblit=True, 
                handle_props={'alpha': 0.3} 
            )
            self.notebook.select(self.single_image_tab)
            self.process_image(1)  
            messagebox.showinfo("ROI Selection", "Draw a rectangle on the image to select ROI")
        else:
            self.roi_selector.set_active(False)
            self.roi_selector = None
            self.canvas.draw()

    def on_roi_select(self, eclick, erelease):
        if self.image is None:
            return
            
        x1, y1 = int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata))
        x2, y2 = int(max(eclick.xdata, erelease.xdata)), int(max(eclick.ydata, erelease.ydata))
        
        height, width = self.image.shape
        x1 = max(0, min(x1, width-1))
        x2 = max(0, min(x2, width-1))
        y1 = max(0, min(y1, height-1))
        y2 = max(0, min(y2, height-1))
        
        self.roi_coords = (x1, y1, x2, y2)
        
        if hasattr(self, 'roi_rect') and self.roi_rect is not None:
            try:
                self.roi_rect.remove()
            except:
                pass
        
        self.roi_rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                    fill=True,
                                    facecolor='yellow',
                                    edgecolor='red',
                                    alpha=0.3,
                                    linewidth=2,
                                    zorder=10)  
        self.ax.add_patch(self.roi_rect)
        self.canvas.draw()
        
        self.notebook.select(self.visualization_tab)
        self.update_histogram()
        self.update_edge_detection()

    def reset_roi(self):
        if self.roi_selector:
            self.roi_selector.set_active(False)
            self.roi_selector = None
        
        if hasattr(self, 'roi_rect') and self.roi_rect is not None:
            try:
                self.roi_rect.remove()
            except:
                pass
            self.roi_rect = None
            self.canvas.draw()
            
        self.roi_coords = None
        self.update_histogram()
        self.update_edge_detection()
        self.process_image(1)

    def save_plot(self):
        if self.image is None:
            messagebox.showwarning("Warning", "No plot to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"),
                      ("JPEG files", "*.jpg"),
                      ("PDF files", "*.pdf"),
                      ("SVG files", "*.svg")]
        )
        
        if file_path:
            try:
                if self.notebook.select() == self.visualization_tab:
                    if self.notebook.index(self.notebook.select()) == 0:
                        self.hist_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    else:
                        self.edge_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", "Plot saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

    def update_histogram(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        if self.roi_coords is not None:
            x1, y1, x2, y2 = self.roi_coords
            img_data = self.image[y1:y2, x1:x2]
        else:
            img_data = self.image
            
        hist, bin_edges = np.histogram(img_data.ravel(), bins=256, range=(0, 256))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        self.hist_ax.clear()
        
        self.hist_ax.bar(bin_centers, hist, width=1.0, color='#2196F3', alpha=0.7,
                        label='Intensity Distribution')
        
        mean = np.mean(img_data)
        median = np.median(img_data)
        std = np.std(img_data)
        
        if self.show_mean_var.get():
            self.hist_ax.axvline(mean, color='#F44336', linestyle='--', linewidth=2,
                               label=f'Mean: {mean:.1f}')
        
        if self.show_median_var.get():
            self.hist_ax.axvline(median, color='#4CAF50', linestyle='--', linewidth=2,
                               label=f'Median: {median:.1f}')
        
        if self.show_std_var.get():
            self.hist_ax.axvline(mean - std, color='#FFC107', linestyle=':', linewidth=2,
                               label=f'Mean ± Std: {std:.1f}')
            self.hist_ax.axvline(mean + std, color='#FFC107', linestyle=':', linewidth=2)
        
        if self.log_scale_var.get():
            self.hist_ax.set_yscale('log')
            self.hist_ax.set_ylim(1, np.max(hist) * 1.1)
        else:
            self.hist_ax.set_yscale('linear')
            self.hist_ax.set_ylim(0, np.max(hist) * 1.1)
        
        self.hist_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        self.hist_ax.set_xlim(-1, 256)
        
        title = "Intensity Distribution"
        if self.roi_coords is not None:
            title += " (ROI)"
        self.hist_ax.set_title(title, pad=10)
        self.hist_ax.set_xlabel("Intensity (ADU)")
        self.hist_ax.set_ylabel("Pixel Count")
        
        self.hist_ax.grid(True, alpha=0.2)
        self.hist_ax.legend(loc='upper right', framealpha=0.8)
        
        self.hist_fig.tight_layout()
        self.hist_canvas.draw()

    def update_edge_detection(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        try:
            low_threshold = int(self.edge_low_threshold_var.get())
            high_threshold = int(self.edge_high_threshold_var.get())
            
            if not (0 <= low_threshold <= 255 and 0 <= high_threshold <= 255):
                raise ValueError("Thresholds must be between 0 and 255")
            if low_threshold >= high_threshold:
                raise ValueError("Lower threshold must be less than upper threshold")
                
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
            
        if self.roi_coords is not None:
            x1, y1, x2, y2 = self.roi_coords
            img_data = self.image[y1:y2, x1:x2]
        else:
            img_data = self.image
            
        edges = cv2.Canny(img_data, low_threshold, high_threshold)
        
        self.edge_ax.clear()
        
        self.edge_ax.imshow(edges, cmap='gray')
        title = f"Edge Detection\nLower Threshold: {low_threshold}, Upper Threshold: {high_threshold}"
        if self.roi_coords is not None:
            title += " (ROI)"
        self.edge_ax.set_title(title)
        self.edge_ax.axis("off")
        
        self.edge_fig.tight_layout()
        self.edge_canvas.draw()

    def show_edge_detection(self):
        self.update_edge_detection()
        
    def show_histogram(self):
        self.update_histogram()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

    plt.close('all') 
