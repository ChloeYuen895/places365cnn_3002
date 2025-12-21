import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# Pillow (PIL) is an external dependency. Provide a friendlier error message
try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
except Exception as e:
    print('\nERROR: Pillow (PIL) is not installed.')
    print('Install it in your environment and re-run this script:')
    print('  python3 -m pip install --user pillow numpy')
    print('\nOn Ubuntu/WSL you may also need Tk bindings:')
    print('  sudo apt update; sudo apt install -y python3-tk')
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
except Exception as e:
    print('\nERROR: PyTorch is not installed.')
    print('Install it in your environment and re-run this script:')
    print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
    sys.exit(1)

import numpy as np

# Paths relative to this script
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / 'models' / 'v2_vgg16_best.pth.tar'

# Your 5 custom classes (in alphabetical order as used during training)
CLASS_NAMES = ['arctic', 'bamboo', 'desert', 'forest', 'grassland']


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')
    
    # Determine device - use CUDA if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained checkpoint to CPU first
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Create VGG16 model with 5 classes
    model = models.vgg16(num_classes=5)
    
    # Handle model state dict loading
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        
        # Check if the state_dict has DataParallel structure (features.module.* keys)
        if any(key.startswith('features.module.') for key in state_dict.keys()):
            # Remove 'features.module.' prefix from keys to match single model structure
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('features.module.'):
                    new_key = key.replace('features.module.', 'features.')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
    else:
        # Direct model state dict
        model.load_state_dict(checkpoint)
    
    # Move model to the appropriate device
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {MODEL_PATH}")
    if 'best_prec1' in checkpoint:
        print(f"Best accuracy: {checkpoint['best_prec1']:.2f}%")
    
    return model


def get_transform():
    """Get the same preprocessing transforms used during training"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


class CustomCNNGUI:
    def __init__(self, root):
        self.root = root
        root.title('Custom 5-Class Places365 Classifier')

        self.model = None
        try:
            self.model = load_model()
            # Get device from model (it's already on the correct device from load_model)
            self.device = next(self.model.parameters()).device
            print(f"Model loaded on device: {self.device}")
        except Exception as e:
            messagebox.showerror('Model load error', str(e))
            root.destroy()
            return

        self.transform = get_transform()

        # Create GUI elements
        self.img_label = tk.Label(root)
        self.img_label.pack(padx=10, pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0,10))

        self.open_btn = tk.Button(btn_frame, text='Open Image', command=self.open_image)
        self.open_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(btn_frame, text='Save Result', state=tk.DISABLED, command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.quit_btn = tk.Button(btn_frame, text='Quit', command=root.quit)
        self.quit_btn.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load an image to classify")
        status_bar = tk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.annotated_img = None
        self.photo = None

    def open_image(self):
        path = filedialog.askopenfilename(
            title='Select image', 
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')]
        )
        if not path:
            return
        try:
            self.status_var.set("Classifying image...")
            self.root.update()
            self.classify_and_show(path)
        except Exception as e:
            messagebox.showerror('Error', str(e))
            self.status_var.set("Error occurred")

    def classify_and_show(self, image_path: str):
        # Load and preprocess image
        pil_img = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        input_tensor = self.transform(pil_img).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(self.device)  # Move to same device as model
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top predictions
        top_prob, top_idx = torch.topk(probabilities, 5)
        top_idx = top_idx.cpu().numpy()
        top_prob = top_prob.cpu().numpy()
        
        # Get top prediction
        predicted_class = CLASS_NAMES[top_idx[0]]
        confidence = top_prob[0]
        
        # Create annotated image
        annotated_pil = pil_img.copy()
        draw = ImageDraw.Draw(annotated_pil)
        
        try:
            font = ImageFont.truetype('arial.ttf', size=24)
            small_font = ImageFont.truetype('arial.ttf', size=16)
        except Exception:
            font = ImageFont.load_default()
            small_font = font
        
        # Main prediction text
        main_text = f'{predicted_class}: {confidence:.1%}'
        
        # Get text size for main prediction
        try:
            bbox = draw.textbbox((0, 0), main_text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = (len(main_text) * 12, 30)
        
        # Draw main prediction
        padding = 10
        bg_coords = [10, 10, 10 + tw + padding * 2, 10 + th + padding * 2]
        draw.rectangle(bg_coords, fill=(0, 0, 0, 200))
        draw.text((10 + padding, 10 + padding), main_text, font=font, fill=(255, 255, 255))
        
        # Draw top 5 predictions in corner
        top5_y = bg_coords[3] + 10
        draw.rectangle([10, top5_y, 250, top5_y + 120], fill=(0, 0, 0, 150))
        
        for i, (idx, prob) in enumerate(zip(top_idx, top_prob)):
            class_name = CLASS_NAMES[idx]
            text = f"{i+1}. {class_name}: {prob:.1%}"
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            draw.text((15, top5_y + 5 + i * 20), text, font=small_font, fill=color)
        
        self.annotated_img = annotated_pil
        self.show_image(annotated_pil)
        self.save_btn.config(state=tk.NORMAL)
        
        # Update status
        self.status_var.set(f"Prediction: {predicted_class} ({confidence:.1%} confidence)")

    def show_image(self, pil_image: Image.Image):
        max_w, max_h = 1000, 800
        w, h = pil_image.size
        scale = min(1.0, max_w / w, max_h / h)
        if scale < 1.0:
            display_img = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            display_img = pil_image
        self.photo = ImageTk.PhotoImage(display_img)
        self.img_label.config(image=self.photo)

    def save_image(self):
        if self.annotated_img is None:
            return
        path = filedialog.asksaveasfilename(
            title='Save annotated image', 
            defaultextension='.jpg', 
            filetypes=[('JPEG', '*.jpg'), ('PNG', '*.png')]
        )
        if not path:
            return
        self.annotated_img.save(path)
        messagebox.showinfo('Saved', f'Annotated image saved to {path}')
        self.status_var.set(f"Image saved to {path}")


def main():
    root = tk.Tk()
    app = CustomCNNGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()