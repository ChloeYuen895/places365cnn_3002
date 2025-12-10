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
import cv2
import numpy as np

# Paths relative to this script
BASE = Path(__file__).resolve().parent
PROTOTXT = BASE / 'models' / 'deploy_vgg16_places365.prototxt'
CAFFEMODEL = BASE / 'models' / 'vgg16_places365.caffemodel'
LABELS_FILE = BASE / 'categories_places365.txt'


def load_model():
    if not PROTOTXT.exists() or not CAFFEMODEL.exists():
        raise FileNotFoundError('Model files missing: put prototxt and caffemodel next to this script')
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))
    return net


def load_labels():
    if LABELS_FILE.exists():
        with open(LABELS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        labels = [ln.split()[0].split('/')[-1].replace('_', ' ') for ln in lines]
        return labels
    else:
        return None


def preprocess_for_caffe(img: np.ndarray):
    h, w = img.shape[:2]
    if h < w:
        new_h = 256
        new_w = int(w * 256 / h)
    else:
        new_w = 256
        new_h = int(h * 256 / w)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (new_h - 224) // 2
    x0 = (new_w - 224) // 2
    crop = resized[y0:y0+224, x0:x0+224]
    blob = cv2.dnn.blobFromImage(crop, scalefactor=1.0, size=(224,224), mean=(104,117,123), swapRB=False, crop=False)
    return blob


class DeployGUI:
    def __init__(self, root):
        self.root = root
        root.title('Places365 Classifier')

        self.net = None
        try:
            self.net = load_model()
        except Exception as e:
            messagebox.showerror('Model load error', str(e))
            root.destroy()
            return

        self.labels = load_labels()

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

        self.annotated_img = None
        self.photo = None

    def open_image(self):
        path = filedialog.askopenfilename(title='Select image', filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff')])
        if not path:
            return
        try:
            self.classify_and_show(path)
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def classify_and_show(self, image_path: str):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise RuntimeError('cv2 failed to read image')

        blob = preprocess_for_caffe(img_bgr)
        self.net.setInput(blob)
        preds = self.net.forward()
        if preds is None:
            raise RuntimeError('model returned no predictions')

        probs = preds.flatten()
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        if self.labels and top_idx < len(self.labels):
            label_name = self.labels[top_idx]
        else:
            label_name = f'index_{top_idx}'

        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        # Draw label text on image (top-left)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', size=20)
        except Exception:
            font = ImageFont.load_default()

        text = f'{label_name}: {top_prob:.3f}'
        # calculate text size and draw rectangle background
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            try:
                tw, th = font.getsize(text)
            except Exception:
                try:
                    tw, th = draw.textsize(text, font=font)
                except Exception:
                    tw, th = (len(text) * 6, 12)
        padding = 6
        draw.rectangle([5, 5, 5 + tw + padding, 5 + th + padding], fill=(0,0,0,160))
        draw.text((8, 8), text, font=font, fill=(255,255,255))

        self.annotated_img = pil
        self.show_image(pil)
        self.save_btn.config(state=tk.NORMAL)

        # prediction popup removed: label is drawn on the image itself

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
        path = filedialog.asksaveasfilename(title='Save annotated image', defaultextension='.jpg', filetypes=[('JPEG', '*.jpg'), ('PNG', '*.png')])
        if not path:
            return
        self.annotated_img.save(path)
        messagebox.showinfo('Saved', f'Annotated image saved to {path}')


def main():
    root = tk.Tk()
    app = DeployGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
