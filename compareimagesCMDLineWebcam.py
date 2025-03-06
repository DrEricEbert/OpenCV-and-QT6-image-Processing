import cv2
import time
import numpy as np
import argparse

def compare_images(image1, image2):
    # Example comparison using Mean Squared Error (MSE)
    mse = np.mean((image1 - image2) ** 2)
    return mse

def capture_webcam_image(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def main():
    parser = argparse.ArgumentParser(description='Image Registration and Comparison')
    parser.add_argument('--use_webcam', action='store_true', help='Use webcam for image capture')
    parser.add_argument('--reg_image', type=str, default='reg_image.jpg', help='Path to registration image file')
    parser.add_argument('--comp_image', type=str, default='comp_image.jpg', help='Path to comparison image file')
    parser.add_argument('--interval', type=int, default=5, help='Interval between captures in seconds')
    args = parser.parse_args()

    if args.use_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not available. Falling back to image files.")
            args.use_webcam = False

    if not args.use_webcam:
        # Load images from files
        reg_image = cv2.imread(args.reg_image)
        comp_image = cv2.imread(args.comp_image)
        if reg_image is None or comp_image is None:
            print("Error: Image files not found.")
            return
        result = compare_images(reg_image, comp_image)
        print(f"MSE between images: {result}")
        return

    # Time-interleaved capture from webcam
    try:
        while True:
            # Capture registration frame
            print(f"Capturing registration image at {time.strftime('%H:%M:%S')}")
            reg_frame = capture_webcam_image(cap)
            if reg_frame is None:
                print("Failed to capture registration frame.")
                break

            # Wait for the interval
            time.sleep(args.interval)

            # Capture comparison frame
            print(f"Capturing comparison image at {time.strftime('%H:%M:%S')}")
            comp_frame = capture_webcam_image(cap)
            if comp_frame is None:
                print("Failed to capture comparison frame.")
                break

            # Process and display results
            result = compare_images(reg_frame, comp_frame)
            print(f"MSE between frames: {result}")

            # Display the captured frames
            cv2.imshow('Registration Frame', reg_frame)
            cv2.imshow('Comparison Frame', comp_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Wait before next cycle
            time.sleep(args.interval)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()