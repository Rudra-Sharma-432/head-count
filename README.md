# Person Counter AI — GitHub Pages

A static browser-based web app for counting people in a photo.

## Features

- Upload or drag & drop a photo.
- Detects `person` objects using TensorFlow.js + COCO-SSD MobileNet v2.
- Optional overlapping-tile mode for wide/high-resolution classroom photos.
- Non-Maximum Suppression removes duplicate detections from overlapping tiles.
- Shows bounding boxes and confidence for each detected person.
- Manual `+` / `-` adjustment.
- No backend and no image upload server.
- Works as a static GitHub Pages website.

## Deploy on GitHub Pages

1. Create a new GitHub repository.
2. Upload `index.html`, `style.css`, and `app.js` to the repository root.
3. Open:
   `Settings → Pages`
4. Under "Build and deployment", choose:
   `Deploy from a branch`
5. Choose the branch containing the files and `/ (root)`.
6. Save.
7. Open the GitHub Pages URL after deployment.

## Important

The app loads TensorFlow.js and the COCO-SSD model from jsDelivr, so the first load requires internet access.

Detection happens in the browser. The uploaded image is not sent to a custom backend by this project.

## Recommended classroom settings

Start with:

- Confidence: 45%
- Wide-photo enhancement: ON

If it misses people, lower confidence gradually to around 30–40%.
If it detects too many false people, raise it toward 50–60%.

## Accuracy limitations

This is object detection, not a guaranteed headcount system. People can be missed when they are very small, heavily occluded, behind objects, outside the model's learned visual patterns, or captured in difficult lighting.

For a very large classroom image, tile mode gives the detector larger effective resolution for distant people, but it increases processing time.

## Technology

- HTML
- CSS
- Vanilla JavaScript
- TensorFlow.js
- COCO-SSD / MobileNet v2

Official model documentation:
https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd
