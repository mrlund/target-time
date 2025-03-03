import cv2
import numpy as np
import argparse
import os
from glob import glob
import time

# Fine-tuning parameters for hole detection
HOLE_MIN_RADIUS = 3  # Minimum radius of a detected hole (pixels)
HOLE_MAX_RADIUS = 10  # Maximum radius of a detected hole (pixels)
HOLE_CONTRAST_THRESHOLD = 50  # Minimum grayscale difference to detect a hole
FRAME_RATE = 2  # Frames per second (used in live mode)
MAX_SHOTS = 10  # Stop after 10 shots
OUTERMOST_RING_RADIUS = 1200  # Radius of the outermost ring (ring 9, score 1)

class TargetScorer:
    def __init__(self, mode="live", image_paths=None):
        self.mode = mode
        self.image_paths = image_paths
        self.camera = None
        
        if self.mode == "live":
            try:
                from picamera2 import Picamera2
                self.camera = Picamera2()
                self.camera.configure(self.camera.create_still_configuration())
                self.camera.start()
            except ImportError:
                raise ImportError("picamera2 is required for live mode. Run in test mode or install picamera2 on a Raspberry Pi.")
        elif self.mode == "test":
            if not image_paths or len(image_paths) < 2:
                raise ValueError("Test mode requires at least two images to compare.")
            self.image_paths.sort()
        
        self.center_x, self.center_y = None, None
        self.black_circle_radius = None
        self.inner_ring_increment = None
        self.outer_ring_increment = None
        self.shots = []
        self.total_score = 0
        self.previous_frame = None
        self.shot_overlay = None
        
    def calibrate_target(self, frame):
        """Calibrate the target by detecting the large black center circle."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Detect the large black circle (diameter ~ half image width)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=200,
            param1=100,
            param2=50,
            minRadius=600,  # Minimum radius for the large black circle
            maxRadius=800   # Maximum radius for the large black circle
        )
        
        # Debug: Draw all detected circles
        debug_frame = frame.copy()
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for i, (x, y, r) in enumerate(circles):
                cv2.circle(debug_frame, (x, y), r, (0, 255, 255), 2)
                cv2.circle(debug_frame, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(debug_frame, f"C{i}", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imwrite("detected_circles.jpg", debug_frame)
            print(f"Saved detected circles image as 'detected_circles.jpg' with {len(circles)} circles detected.")

            # Select the circle closest to the image center
            image_center_x, image_center_y = frame.shape[1] // 2, frame.shape[0] // 2
            distances = [np.sqrt((x - image_center_x)**2 + (y - image_center_y)**2) for x, y, _ in circles]
            closest_circle_idx = np.argmin(distances)
            self.center_x, self.center_y, self.black_circle_radius = circles[closest_circle_idx]
            
            # Calculate ring increments
            # Inner rings (0-3, scores 10-7) within the black circle
            self.inner_ring_increment = self.black_circle_radius / 3
            # Outer rings (4-9, scores 6-1) from black circle edge to outermost ring
            self.outer_ring_increment = (OUTERMOST_RING_RADIUS - self.black_circle_radius) / 6
            
            print(f"Calibrated: Center at ({self.center_x}, {self.center_y}), Black circle radius: {self.black_circle_radius}")
            print(f"Inner ring increment: {self.inner_ring_increment}, Outer ring increment: {self.outer_ring_increment}")
        else:
            raise ValueError("Could not detect the black center circle for calibration.")
    
    def draw_scoring_zones(self, frame):
        """Draw the scoring zones (rings 0-9, scores 10-1) on the frame."""
        # Inner rings (0-3, scores 10-7)
        for ring_number in range(4):
            radius = int(self.inner_ring_increment * ring_number)
            score = 10 - ring_number
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][ring_number % 3]
            cv2.circle(frame, (self.center_x, self.center_y), radius, color, 2)
            label_x = self.center_x
            label_y = self.center_y - radius - 20
            cv2.putText(frame, str(score), (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Outer rings (4-9, scores 6-1)
        for ring_number in range(4, 10):
            distance_from_center = self.black_circle_radius + (ring_number - 3) * self.outer_ring_increment
            radius = int(distance_from_center)
            score = 10 - ring_number
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][ring_number % 3]
            cv2.circle(frame, (self.center_x, self.center_y), radius, color, 2)
            label_x = self.center_x
            label_y = self.center_y - radius - 20
            cv2.putText(frame, str(score), (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print("Drew scoring zones on the output frame.")

    def score_shot(self, x, y):
        """Calculate the score of a shot based on its position."""
        distance = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        
        if distance <= self.black_circle_radius:
            # Inside the black circle (rings 0-3, scores 10-7)
            ring_number = int(distance / self.inner_ring_increment)
            score = 10 - ring_number
        else:
            # Outside the black circle (rings 4-9, scores 6-1)
            distance_from_edge = distance - self.black_circle_radius
            ring_number = 3 + int(distance_from_edge / self.outer_ring_increment)
            if ring_number <= 9:
                score = 10 - ring_number
            else:
                score = 0
        
        print(f"Scoring shot at ({x}, {y}): Distance = {distance:.1f}, Ring number = {ring_number}, Score = {score}")
        return score
    
    def detect_new_hole(self, current_frame):
        """Detect a new bullet hole by comparing the current frame with the previous one."""
        if self.previous_frame is None:
            return None
        
        prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, HOLE_CONTRAST_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if HOLE_MIN_RADIUS <= radius <= HOLE_MAX_RADIUS:
                return int(x), int(y)
        
        return None
    
    def run(self):
        """Main loop to process frames (live or test mode), detect shots, and score them."""
        print(f"Starting target scorer in {self.mode} mode...")
        
        if self.mode == "live":
            frame = self.camera.capture_array()
        else:
            frame = cv2.imread(self.image_paths[0])
            if frame is None:
                raise ValueError(f"Could not load image: {self.image_paths[0]}")
        
        self.calibrate_target(frame)
        self.previous_frame = frame.copy()
        output_frame = frame.copy()
        self.shot_overlay = np.zeros_like(output_frame, dtype=np.uint8)
        
        h, w = frame.shape[:2]
        print(f"Image dimensions: {w} x {h}")
        print(f"Output frame shape: {output_frame.shape}, dtype: {output_frame.dtype}")
        print(f"Shot overlay shape: {self.shot_overlay.shape}, dtype: {self.shot_overlay.dtype}")
        
        self.draw_scoring_zones(output_frame)
        success = cv2.imwrite("scoring_zones.jpg", output_frame)
        if success:
            print("Saved image with scoring zones as 'scoring_zones.jpg'")
        else:
            print("Failed to save image with scoring zones")
        
        cv2.circle(output_frame, (self.center_x, self.center_y), 30, (255, 255, 0), -1)
        cv2.imwrite("scoring_zones_with_test.jpg", output_frame)
        print("Saved scoring_zones_with_test.jpg with a test marker at the center")
        
        shot_count = 0
        image_index = 1 if self.mode == "test" else 0
        
        while shot_count < MAX_SHOTS:
            if self.mode == "live":
                frame = self.camera.capture_array()
            else:
                if image_index >= len(self.image_paths):
                    print("No more images to process.")
                    break
                frame = cv2.imread(self.image_paths[image_index])
                if frame is None:
                    print(f"Could not load image: {self.image_paths[image_index]}")
                    break
                image_index += 1
            
            new_hole = self.detect_new_hole(frame)
            
            if new_hole is not None:
                x, y = new_hole
                shot_count += 1
                
                score = self.score_shot(x, y)
                self.total_score += score
                self.shots.append((x, y, score))
                
                print(f"Shot {shot_count}: Position ({x}, {y}), Score: {score}, Total: {self.total_score}")
                
                print(f"Drawing shot {shot_count} at ({x}, {y}) on shot overlay")
                cv2.circle(self.shot_overlay, (x, y), 20, (0, 0, 255), -1)
                cv2.putText(self.shot_overlay, str(shot_count), (x + 30, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
                
                # Debug: Save the shot overlay directly
                overlay_filename = f"shot_overlay_{shot_count}.jpg"
                success = cv2.imwrite(overlay_filename, self.shot_overlay)
                if success:
                    print(f"Saved shot overlay after shot {shot_count}: {overlay_filename}")
                else:
                    print(f"Failed to save shot overlay after shot {shot_count}")
                
                # Combine using cv2.add
                combined_frame = cv2.add(output_frame, self.shot_overlay)
                intermediate_filename = f"intermediate_shot_{shot_count}.jpg"
                success = cv2.imwrite(intermediate_filename, combined_frame)
                if success:
                    print(f"Saved intermediate image after shot {shot_count}: {intermediate_filename}")
                else:
                    print(f"Failed to save intermediate image after shot {shot_count}")
            
            self.previous_frame = frame.copy()
            
            if self.mode == "live":
                time.sleep(1.0 / FRAME_RATE)
            else:
                time.sleep(0.5)
        
        output_filename = f"target_result_{int(time.time())}.jpg"
        print("Saving output image...")
        combined_frame = cv2.add(output_frame, self.shot_overlay)
        success = cv2.imwrite(output_filename, combined_frame)
        if success:
            print(f"Session complete! Total score: {self.total_score}. Output saved as '{output_filename}'.")
        else:
            print(f"Failed to save output image '{output_filename}'.")
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(test_image, (50, 50), 10, (255, 0, 0), -1)
        cv2.imwrite("test_drawing.jpg", test_image)
        print("Saved test_drawing.jpg to verify OpenCV drawing functionality.")

        if self.mode == "live" and self.camera is not None:
            self.camera.stop()

def parse_args():
    parser = argparse.ArgumentParser(description="Target scoring program with live and test modes.")
    parser.add_argument("--mode", choices=["live", "test"], default="live",
                        help="Mode to run in: 'live' for camera, 'test' for images")
    parser.add_argument("--images", nargs="+",
                        help="Paths to images for test mode (e.g., '*.jpg' or 'img1.jpg img2.jpg')")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    image_paths = []
    if args.mode == "test":
        if not args.images:
            raise ValueError("Test mode requires image paths. Use --images <paths>.")
        for path in args.images:
            expanded = glob(path)
            if expanded:
                image_paths.extend(expanded)
            else:
                image_paths.append(path)
    
    scorer = TargetScorer(mode=args.mode, image_paths=image_paths if args.mode == "test" else None)
    try:
        scorer.run()
    except KeyboardInterrupt:
        if scorer.mode == "live" and scorer.camera is not None:
            scorer.camera.stop()
        print("Program terminated by user.")