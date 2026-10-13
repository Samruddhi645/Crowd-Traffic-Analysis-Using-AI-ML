import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import numpy as np
from collections import Counter
import os

model = YOLO("best.pt")  

vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "van", "pedestrian"}

class ModernButton(tk.Button):
    def __init__(self, parent, **kwargs):
        self.hover_bg = kwargs.pop('hover_bg', '#2980b9')
        
        default_config = {
            'font': ('Segoe UI', 12, 'bold'),
            'relief': 'flat',
            'bd': 0,
            'cursor': 'hand2',
            'padx': 25,
            'pady': 12
        }
        default_config.update(kwargs)
        super().__init__(parent, **default_config)
        self.default_bg = default_config.get('bg', '#3498db')
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_enter(self, event):
        self.config(bg=self.hover_bg)
    
    def _on_leave(self, event):
        self.config(bg=self.default_bg)

class StatusCard(tk.Frame):
    def __init__(self, parent, title, icon="", bg_color="#2c3e50"):
        super().__init__(parent, bg=bg_color, relief='solid', bd=1)
        
        header_frame = tk.Frame(self, bg=bg_color)
        header_frame.pack(fill="x", padx=15, pady=(15, 5))
        
        title_label = tk.Label(
            header_frame, 
            text=f"{icon} {title}", 
            font=('Segoe UI', 11, 'bold'), 
            fg='#bdc3c7', 
            bg=bg_color
        )
        title_label.pack(anchor="w")
        
        self.value_label = tk.Label(
            self, 
            font=('Segoe UI', 16, 'bold'), 
            bg=bg_color,
            pady=10
        )
        self.value_label.pack(fill="x", padx=15, pady=(0, 15))
    
    def update_status(self, text, color='#ecf0f1'):
        self.value_label.config(text=text, fg=color)

class VehicleCountCard(tk.Frame):
    def __init__(self, parent, bg_color="#2c3e50"):
        super().__init__(parent, bg=bg_color, relief='solid', bd=1)
        
        header_frame = tk.Frame(self, bg=bg_color)
        header_frame.pack(fill="x", padx=15, pady=(15, 5))
        
        title_label = tk.Label(
            header_frame, 
            text="🚗 Vehicle Count", 
            font=('Segoe UI', 11, 'bold'), 
            fg='#bdc3c7', 
            bg=bg_color
        )
        title_label.pack(anchor="w")
        
        canvas = tk.Canvas(self, bg=bg_color, highlightthickness=0, height=200)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg=bg_color)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=15, pady=(0, 15))
        scrollbar.pack(side="right", fill="y", pady=(0, 15))
        
        self.vehicle_labels = {}
        self.canvas = canvas
    
    def update_counts(self, vehicle_counts):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.vehicle_labels.clear()
        
        for vehicle_type, count in vehicle_counts.items():
            count_frame = tk.Frame(self.scrollable_frame, bg="#2c3e50")
            count_frame.pack(fill="x", pady=2)
            
            icons = {
                "car": "🚗", "truck": "🚛", "bus": "🚌", 
                "motorcycle": "🏍️", "bicycle": "🚴", 
                "van": "🚐", "pedestrian": "🚶"
            }
            icon = icons.get(vehicle_type, "🚗")
            
            vehicle_label = tk.Label(
                count_frame,
                text=f"{icon} {vehicle_type.capitalize()}: {count}",
                font=('Segoe UI', 12),
                fg='#ecf0f1',
                bg="#2c3e50",
                anchor="w"
            )
            vehicle_label.pack(fill="x")
            
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

class TrafficAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚦 Traffic Analysis System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#34495e")
        self.root.resizable(True, True)
        
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.create_header()
        self.create_main_content()
        self.create_controls()
        self.create_footer()

        self.cap = None
        self.running = False
        self.current_file = None
        self.vehicle_counts = Counter()

    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=100)
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 0))
        header_frame.grid_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🚦 Traffic Analysis System",
            font=("Segoe UI", 28, "bold"),
            fg="#e67e22",
            bg="#2c3e50"
        )
        title_label.pack(expand=True, pady=(20, 5))
        
        
        subtitle_label = tk.Label(
            header_frame,
            text="AI-Powered Vehicle Detection and Traffic Analysis",
            font=("Segoe UI", 12),
            fg='#bdc3c7',
            bg="#2c3e50"
        )
        subtitle_label.pack(pady=(0, 15))

    def create_main_content(self):
        main_frame = tk.Frame(self.root, bg="#34495e")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_columnconfigure(0, weight=1)

        self.create_status_panel(main_frame)
        
        self.create_display_panel(main_frame)

    def create_status_panel(self, parent):
        status_frame = tk.Frame(parent, bg="#34495e")
        status_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        
        self.system_status_card = StatusCard(status_frame, "System Status", "🔧")
        self.system_status_card.pack(fill="x", pady=(0, 15))
        self.system_status_card.update_status("System Ready", "#27ae60")
        
        self.detection_card = StatusCard(status_frame, "Detection Status", "🔍")
        self.detection_card.pack(fill="x", pady=(0, 15))
        self.detection_card.update_status("Waiting for input...", "#95a5a6")
        
        self.total_vehicles_card = StatusCard(status_frame, "Total Vehicles", "📊")
        self.total_vehicles_card.pack(fill="x", pady=(0, 15))
        self.total_vehicles_card.update_status("0", "#95a5a6")

        self.vehicle_count_card = VehicleCountCard(status_frame)
        self.vehicle_count_card.pack(fill="both", expand=True, pady=(15, 0))

    def create_display_panel(self, parent):
        display_frame = tk.Frame(parent, bg="#2c3e50", relief='solid', bd=2)
        display_frame.grid(row=0, column=1, sticky="nsew")
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        
        display_header = tk.Label(
            display_frame,
            text="📹 Live Feed / Image Analysis",
            font=("Segoe UI", 16, "bold"),
            fg="#ecf0f1",
            bg="#2c3e50"
        )
        display_header.grid(row=0, column=0, pady=(20, 10))
        
        self.display_label = tk.Label(
            display_frame,
            bg="#34495e",
            text="📷\n\nCamera feed or uploaded image will appear here\n\nUse controls below to start detection or upload files",
            font=("Segoe UI", 14),
            fg="#bdc3c7",
            justify="center"
        )
        self.display_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

    def create_controls(self):
        control_frame = tk.Frame(self.root, bg="#34495e")
        control_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        
        button_row1 = tk.Frame(control_frame, bg="#34495e")
        button_row1.pack(pady=(0, 10))
        
        button_row2 = tk.Frame(control_frame, bg="#34495e")
        button_row2.pack()
        
        self.start_camera_button = ModernButton(
            button_row1,
            text="📹 Start Camera",
            command=self.start_camera_detection,
            bg="#27ae60",
            hover_bg="#229954",
            fg="white",
            width=15
        )
        self.start_camera_button.pack(side="left", padx=(0, 15))
        
        self.stop_camera_button = ModernButton(
            button_row1,
            text="⏹ Stop Camera",
            command=self.stop_camera_detection,
            bg="#e74c3c",
            hover_bg="#c0392b",
            fg="white",
            width=15
        )
        self.stop_camera_button.pack(side="left", padx=(0, 15))
        

        self.upload_button = ModernButton(
            button_row2,
            text="📁 Upload Image/Video",
            command=self.upload_file,
            bg="#3498db",
            hover_bg="#2980b9",
            fg="white",
            width=18
        )
        self.upload_button.pack(side="left", padx=(0, 15))
        
        self.analyze_file_button = ModernButton(
            button_row2,
            text="🔍 Analyze File",
            command=self.analyze_file,
            bg="#9b59b6",
            hover_bg="#8e44ad",
            fg="white",
            width=15
        )
        self.analyze_file_button.pack(side="left", padx=(0, 15))
        
        self.clear_button = ModernButton(
            button_row2,
            text="🗑 Clear",
            command=self.clear_display,
            bg="#95a5a6",
            hover_bg="#7f8c8d",
            fg="white",
            width=10
        )
        self.clear_button.pack(side="left")

    def create_footer(self):
        footer_frame = tk.Frame(self.root, bg="#2c3e50", height=50)
        footer_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 20))
        footer_frame.grid_propagate(False)
        
        footer_label = tk.Label(
            footer_frame,
            text="🤖 Powered by YOLO AI Model | Real-time Traffic Analysis | Drone Imagery Support | Version 1.0",
            font=("Segoe UI", 10),
            fg="#bdc3c7",
            bg="#2c3e50"
        )
        footer_label.pack(expand=True, pady=15)

    def start_camera_detection(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access camera. Please check camera connection.")
            self.running = False
            return
            
        self.system_status_card.update_status("🔴 Camera Active", "#e74c3c")
        self.detection_card.update_status("Analyzing traffic...", "#f39c12")
        self.vehicle_counts = Counter()
        self.update_camera_frame()

    def stop_camera_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
        
        self.display_label.config(
            image='',
            text="📷\n\nCamera feed stopped\n\nUse controls below to start detection or upload files",
            fg="#bdc3c7"
        )
        
        self.system_status_card.update_status("System Ready", "#27ae60")
        self.detection_card.update_status("Camera Stopped", "#e74c3c")

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video File",
            filetypes=[
                ("All Supported", "*.jpg *.jpeg *.png *.bmp *.tiff *.mp4 *.avi *.mov *.mkv"),
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.current_file = file_path
            filename = os.path.basename(file_path)
            self.system_status_card.update_status(f"File Loaded: {filename}", "#3498db")
            self.detection_card.update_status("File ready for analysis", "#f39c12")
            
            # Display the file (for images)
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                self.display_image(file_path)

    def display_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image to fit display
            height, width = img.shape[:2]
            max_width, max_height = 700, 500
            
            if width > max_width or height > max_height:
                aspect_ratio = width / height
                if width > height:
                    new_width = max_width
                    new_height = int(max_width / aspect_ratio)
                else:
                    new_height = max_height
                    new_width = int(max_height * aspect_ratio)
                
                img = cv2.resize(img, (new_width, new_height))
            
            img_pil = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.display_label.imgtk = img_pil
            self.display_label.configure(image=img_pil, text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {str(e)}")

    def analyze_file(self):
        if not self.current_file:
            messagebox.showwarning("No File", "Please upload a file first.")
            return
        
        try:
            if self.current_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                self.analyze_image()
            elif self.current_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.analyze_video()
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error analyzing file: {str(e)}")

    def analyze_image(self):
        img = cv2.imread(self.current_file)
        results = model(img)
        
        # Count vehicles
        self.vehicle_counts = Counter()
        
        # Process detections
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id].lower()
                    confidence = float(box.conf)
                    
                    if confidence > 0.5:  # Confidence threshold
                        self.vehicle_counts[class_name] += 1
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{class_name}: {confidence:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update display
        self.update_detection_results(img)

    def analyze_video(self):
        # For video analysis, we'll process frames and show the result
        cap = cv2.VideoCapture(self.current_file)
        self.vehicle_counts = Counter()
        frame_count = 0
        
        self.detection_card.update_status("Processing video...", "#f39c12")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            # Process every 30th frame to speed up analysis
            if frame_count % 30 == 0:
                results = model(frame)
                
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            class_name = model.names[class_id].lower()
                            confidence = float(box.conf)
                            
                            if confidence > 0.5:
                                self.vehicle_counts[class_name] += 1
        
        cap.release()
        
        # Show final frame with detections
        cap = cv2.VideoCapture(self.current_file)
        ret, final_frame = cap.read()
        if ret:
            results = model(final_frame)
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf)
                        if confidence > 0.5:
                            class_id = int(box.cls)
                            class_name = model.names[class_id].lower()
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(final_frame, f"{class_name}: {confidence:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.update_detection_results(final_frame)
        
        cap.release()

    def update_detection_results(self, frame):
        # Convert and display frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit display
        height, width = img.shape[:2]
        max_width, max_height = 700, 500
        
        if width > max_width or height > max_height:
            aspect_ratio = width / height
            if width > height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            
            img = cv2.resize(img, (new_width, new_height))
        
        img_pil = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.display_label.imgtk = img_pil
        self.display_label.configure(image=img_pil, text="")
        
        # Update status cards
        total_vehicles = sum(self.vehicle_counts.values())
        self.total_vehicles_card.update_status(str(total_vehicles), "#3498db")
        self.detection_card.update_status("Analysis Complete", "#27ae60")
        self.vehicle_count_card.update_counts(dict(self.vehicle_counts))

    def update_camera_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_camera_frame)
            return

        results = model(frame)
        
        # Count vehicles in current frame
        current_frame_counts = Counter()
        
        # Process detections
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id].lower()
                    confidence = float(box.conf)
                    
                    if confidence > 0.5:  # Confidence threshold
                        current_frame_counts[class_name] += 1
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update running totals (you might want to implement a more sophisticated counting logic)
        for vehicle_type, count in current_frame_counts.items():
            if count > self.vehicle_counts.get(vehicle_type, 0):
                self.vehicle_counts[vehicle_type] = count

        # Convert and display frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (700, 500))
        img_pil = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.display_label.imgtk = img_pil
        self.display_label.configure(image=img_pil, text="")

        # Update status
        total_vehicles = sum(self.vehicle_counts.values())
        self.total_vehicles_card.update_status(str(total_vehicles), "#3498db")
        self.detection_card.update_status("🔍 Live Analysis", "#27ae60")
        self.vehicle_count_card.update_counts(dict(self.vehicle_counts))

        self.root.after(10, self.update_camera_frame)

    def clear_display(self):
        self.stop_camera_detection()
        self.current_file = None
        self.vehicle_counts = Counter()
        
        self.display_label.config(
            image='',
            text="📷\n\nCamera feed or uploaded image will appear here\n\nUse controls below to start detection or upload files",
            fg="#bdc3c7"
        )
        
        self.system_status_card.update_status("System Ready", "#27ae60")
        self.detection_card.update_status("Waiting for input...", "#95a5a6")
        self.total_vehicles_card.update_status("0", "#95a5a6")
        self.vehicle_count_card.update_counts({})

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficAnalysisApp(root)
    root.mainloop()