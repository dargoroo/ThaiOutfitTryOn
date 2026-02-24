import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageTk, ImageDraw

class OutfitEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Outfit Standardizer (เครื่องมือจัดตำแหน่งชุด)")
        
        # Standard Canvas Size (Logical)
        self.canvas_w = 800
        self.canvas_h = 1200
        
        self.view_zoom = 0.6  # Default zoom level for viewing (60%)
        
        # Default State
        self.current_img = None       # Original PIL Image (Cropped)
        self.display_img = None       # Scaled PIL Image (Logical Size)
        self.tk_img = None            # PhotoImage for Canvas (View Size)
        
        self.img_x = self.canvas_w // 2
        self.img_y = self.canvas_h // 2
        self.scale = 1.0
        self.width_ratio = 1.0        # Ratio to stretch/shrink width
        self.height_ratio = 1.0       # Ratio to stretch/shrink height
        
        # Dragging State
        self.drag_data = {"x": 0, "y": 0, "dragging": False}
        
        self.setup_ui()
        self.draw_guidelines()

    def setup_ui(self):
        # ---------------- Control Panel ----------------
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Button(control_frame, text="1. เปิดรูปชุด (Open Image)", command=self.load_image).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="2. บันทึกรูป (Save Image)", command=self.save_image).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        ttk.Label(control_frame, text="ปรับขนาดชุดรวม (Scale):").pack(anchor=tk.W)
        self.scale_slider = ttk.Scale(control_frame, from_=0.1, to=3.0, value=1.0, orient=tk.HORIZONTAL, command=self.on_scale)
        self.scale_slider.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="ปรับความกว้าง (Width):", foreground="#006400").pack(anchor=tk.W)
        self.width_slider = ttk.Scale(control_frame, from_=0.5, to=2.0, value=1.0, orient=tk.HORIZONTAL, command=self.on_width)
        self.width_slider.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="ปรับความยาว (Height):", foreground="#8B0000").pack(anchor=tk.W)
        self.height_slider = ttk.Scale(control_frame, from_=0.5, to=2.0, value=1.0, orient=tk.HORIZONTAL, command=self.on_height)
        self.height_slider.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="คุณสามารถ 'คลิกแล้วลาก' ที่รูป\nเพื่อเลื่อนตำแหน่งได้เลย", foreground="gray").pack(anchor=tk.W, pady=10)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="ซูมมุมมอง (View Zoom):", foreground="blue").pack(anchor=tk.W)
        self.zoom_slider = ttk.Scale(control_frame, from_=0.3, to=1.5, value=self.view_zoom, orient=tk.HORIZONTAL, command=self.on_view_zoom)
        self.zoom_slider.pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        ttk.Label(control_frame, text="เส้นนำสายตา (Guidelines)", font=(None, 10, 'bold')).pack(anchor=tk.W)
        ttk.Label(control_frame, text="- เส้นแดง: ระดับคอ (Neck)").pack(anchor=tk.W)
        ttk.Label(control_frame, text="- เส้นน้ำเงิน: กึ่งกลางลำตัว").pack(anchor=tk.W)
        ttk.Label(control_frame, text="- เส้นเขียว: ลำตัดไหล่โดยประมาณ").pack(anchor=tk.W)
        ttk.Label(control_frame, text="- เส้นเหลือง: ระดับข้อเท้า/พื้น").pack(anchor=tk.W)
        
        # ---------------- Canvas ----------------
        self.canvas = tk.Canvas(self.root, width=int(self.canvas_w * self.view_zoom), height=int(self.canvas_h * self.view_zoom), bg='white', cursor="fleur")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Bind Mouse Events for Canvas
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_stop)

    def draw_guidelines(self):
        self.canvas.delete("guideline")
        z = self.view_zoom
        # Center Vertical (Blue)
        self.canvas.create_line(int(self.canvas_w//2 * z), 0, int(self.canvas_w//2 * z), int(self.canvas_h * z), fill="blue", dash=(4, 4), tags="guideline")
        # Neck Line at 10% (Red)
        neck_y = int(self.canvas_h * 0.10 * z)
        self.canvas.create_line(0, neck_y, int(self.canvas_w * z), neck_y, fill="red", dash=(4, 4), width=2, tags="guideline")
        # Shoulder Box (Green) - Approximate
        shoulder_y = neck_y + int(40 * z)
        self.canvas.create_line(int((self.canvas_w//2 - 150) * z), shoulder_y, int((self.canvas_w//2 + 150) * z), shoulder_y, fill="green", dash=(2, 4), tags="guideline")
        # Ankle/Floor Line at 95% (Yellow)
        bottom_y = int(self.canvas_h * 0.95 * z)
        self.canvas.create_line(0, bottom_y, int(self.canvas_w * z), bottom_y, fill="orange", dash=(4, 4), width=2, tags="guideline")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="เลือกรูปชุด", 
            filetypes=[("PNG Images", "*.png"), ("All Files", "*.*")]
        )
        if not file_path:
            return
            
        try:
            img = Image.open(file_path).convert("RGBA")
            # Auto-crop transparent borders
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
                
            self.current_img = img
            self.scale = 1.0
            self.width_ratio = 1.0
            self.height_ratio = 1.0
            
            self.scale_slider.set(1.0)
            self.width_slider.set(1.0)
            self.height_slider.set(1.0)
            
            # Reset position to center initially
            self.img_x = self.canvas_w // 2
            # Align top of image to neck line (10% height) roughly
            self.img_y = int(self.canvas_h * 0.10) + (self.current_img.height // 2)
            
            self.update_canvas()
        except Exception as e:
            messagebox.showerror("Error", f"ไม่สามารถโหลดรูปได้: {str(e)}")

    def on_scale(self, val):
        if self.current_img:
            self.scale = float(val)
            self.update_canvas()

    def on_width(self, val):
        if self.current_img:
            self.width_ratio = float(val)
            self.update_canvas()
            
    def on_height(self, val):
        if self.current_img:
            self.height_ratio = float(val)
            self.update_canvas()

    def on_view_zoom(self, val):
        self.view_zoom = float(val)
        self.canvas.config(width=int(self.canvas_w * self.view_zoom), height=int(self.canvas_h * self.view_zoom))
        self.draw_guidelines()
        if self.current_img:
            self.update_canvas()

    def update_canvas(self):
        if not self.current_img:
            return
            
        # Apply standard scale first, then apply individual stretch ratios
        new_w = int(self.current_img.width * self.scale * self.width_ratio)
        new_h = int(self.current_img.height * self.scale * self.height_ratio)
        
        if new_w <= 0 or new_h <= 0:
            return
            
        self.display_img = self.current_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Scale for view
        view_w = max(1, int(new_w * self.view_zoom))
        view_h = max(1, int(new_h * self.view_zoom))
        view_img = self.display_img.resize((view_w, view_h), Image.Resampling.BILINEAR)
        
        self.tk_img = ImageTk.PhotoImage(view_img)
        
        self.canvas.delete("outfit")
        # Draw image
        self.image_id = self.canvas.create_image(
            int(self.img_x * self.view_zoom), int(self.img_y * self.view_zoom), 
            image=self.tk_img, 
            tags="outfit"
        )
        # Guidelines should stay on top
        self.canvas.tag_raise("guideline")

    def on_drag_start(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.drag_data["dragging"] = True

    def on_drag_motion(self, event):
        if self.drag_data["dragging"] and self.current_img:
            dx = (event.x - self.drag_data["x"]) / self.view_zoom
            dy = (event.y - self.drag_data["y"]) / self.view_zoom
            
            self.img_x += dx
            self.img_y += dy
            
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            
            self.update_canvas()

    def on_drag_stop(self, event):
        self.drag_data["dragging"] = False

    def save_image(self):
        if not self.current_img or not self.display_img:
            messagebox.showwarning("Warning", "กรุณาโหลดรูปภาพก่อนบันทึก")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="บันทึกรูปชุด",
            defaultextension=".png",
            filetypes=[("PNG Images", "*.png")]
        )
        if not save_path:
            return
            
        try:
            # Create a blank transparent canvas
            final_img = Image.new("RGBA", (self.canvas_w, self.canvas_h), (0, 0, 0, 0))
            
            # Calculate paste coordinates (top-left)
            paste_x = int(self.img_x - (self.display_img.width / 2))
            paste_y = int(self.img_y - (self.display_img.height / 2))
            
            # Support pasting outside bounds cleanly
            final_img.paste(self.display_img, (paste_x, paste_y))
            
            final_img.save(save_path, "PNG")
            messagebox.showinfo("Success", f"บันทึกรูปเสร็จสิ้น!\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"เกิดข้อผิดพลาดในการบันทึก: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Make it a bit more responsive and adjust screen center
    window_width = 1100
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_cordinate = int((screen_width/2) - (window_width/2))
    y_cordinate = int((screen_height/2) - (window_height/2))
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
    
    app = OutfitEditor(root)
    root.mainloop()
