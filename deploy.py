import cv2
from cv2 import dnn
from pathlib import Path
import sys
import numpy as np

# Build robust paths relative to this script so it works from any CWD
BASE = Path(__file__).resolve().parent
PROTOTXT = BASE / 'deploy_vgg16_places365.prototxt'
CAFFEMODEL = BASE / 'vgg16_places365.caffemodel'
IMAGE_PATH = BASE / 'zoo5' / 'train' / 'arctic' / 'Places365_val_00000897.jpg'

print('PROTOTXT ->', PROTOTXT)
print('CAFFEMODEL ->', CAFFEMODEL)
print('IMAGE_PATH ->', IMAGE_PATH)
print('IMAGE exists?', IMAGE_PATH.exists())

if not PROTOTXT.exists() or not CAFFEMODEL.exists():
	print('Model files not found. Check that prototxt and caffemodel are present.')
	sys.exit(1)

model = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(CAFFEMODEL))

# Try reading the image and print a helpful diagnostic if it fails
image = cv2.imread(str(IMAGE_PATH))
if image is None:
	alt = str(IMAGE_PATH).replace('\\\\', '/')
	image = cv2.imread(alt)
	if image is None:
		print('ERROR: cv2.imread failed to load image. Tried paths:')
		print(' -', str(IMAGE_PATH))
		print(' -', alt)
		sys.exit(1)
	else:
		print('Loaded image using alternative path:', alt)
else:
	print('Loaded image:', IMAGE_PATH)

print('Image shape:', image.shape)

# Prepare input for Caffe VGG16-style model (Places365-trained)
def preprocess_for_caffe(img):
	# Resize shorter side to 256, then center-crop 224x224
	h, w = img.shape[:2]
	if h < w:
		new_h = 256
		new_w = int(w * 256 / h)
	else:
		new_w = 256
		new_h = int(h * 256 / w)
	resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
	# center crop
	y0 = (new_h - 224) // 2
	x0 = (new_w - 224) // 2
	crop = resized[y0:y0+224, x0:x0+224]
	# Caffe models expect BGR with mean subtraction (approx): [104,117,123]
	blob = cv2.dnn.blobFromImage(crop, scalefactor=1.0, size=(224,224), mean=(104,117,123), swapRB=False, crop=False)
	return blob

blob = preprocess_for_caffe(image)
model.setInput(blob)
preds = model.forward()
if preds is None:
	print('ERROR: model.forward() returned None')
	sys.exit(1)

# preds shape: (1, N). Get top-5 indices
probs = preds.flatten()
top_idxs = np.argsort(probs)[::-1][:5]

# Try to load Places365 category labels from remote if not present locally
LABELS_FILE = BASE / 'categories_places365.txt'
labels = None
if LABELS_FILE.exists():
	with open(LABELS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
		lines = [ln.strip() for ln in f if ln.strip()]
		# File format: "/category/name index" (name first, index second)
		# We take the category token, use the last path component, and make it human-readable
		labels = [ln.split()[0].split('/')[-1].replace('_', ' ') for ln in lines]
else:
	try:
		import urllib.request
		url = 'https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt'
		print('Downloading labels from', url)
		with urllib.request.urlopen(url, timeout=10) as resp:
			data = resp.read().decode('utf-8')
		lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
		labels = [ln.split()[0].split('/')[-1].replace('_', ' ') for ln in lines]
		# save a local copy for future runs
		try:
			with open(LABELS_FILE, 'w', encoding='utf-8') as f:
				f.write('\n'.join(lines))
			print('Saved labels to', LABELS_FILE)
		except Exception:
			pass
	except Exception as e:
		print('Could not download labels:', e)

print('\nTop predictions:')
for i in top_idxs:
	prob = float(probs[i])
	name = labels[i] if labels and i < len(labels) else f'index_{i}'
	print(f' - {name}: {prob:.5f}')

print('\nClassification completed successfully.')