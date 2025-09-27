import jetson.inference
import jetson.utils


net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

input_image_path = "/home/nvidia/Desktop/my_images/person1.jpg"

# Load the image into GPU memory
img = jetson.utils.loadImage(input_image_path)

# Perform object detection
# detections is a list containing information about all detected objects
detections = net.Detect(img)

# Print detection results to terminal
print(f"=== Detection Results for {input_image_path} ===")
print(f"Number of objects detected: {len(detections)}")

# Loop through each detected object
for i, detection in enumerate(detections):
    print(f"\n--- Object #{i+1} ---")
    print(f"  ClassID: {detection.ClassID}")
    print(f"  Confidence: {detection.Confidence:.4f}")
    print(f"  Left, Top: ({detection.Left:.2f}, {detection.Top:.2f})")
    print(f"  Right, Bottom: ({detection.Right:.2f}, {detection.Bottom:.2f})")
    print(f"  Width: {detection.Width:.2f}")
    print(f"  Height: {detection.Height:.2f}")
    print(f"  Area: {detection.Area:.2f}")
    print(f"  Center: ({detection.CenterX:.2f}, {detection.CenterY:.2f})")

# Optional: Save the image with bounding boxes to a new file
output_image_path = "/home/nvidia/Desktop/my_images/image1_result.jpg"
jetson.utils.saveImage(output_image_path, img)
print(f"\nImage with bounding boxes saved to: {output_image_path}")

print("\nDetection completed!")
