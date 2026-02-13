import cv2
import pyvirtualcam
import numpy as np
import time

# --- Configuration ---
ASCII_CHARS = " .:-=+*#%@"  
FONT_SIZE = 10              
GRID_WIDTH = 80             
ENABLE_COLOR = True         

def pixel_to_ascii(image, width=GRID_WIDTH):
    height, orig_width, _ = image.shape
    aspect_ratio = height / orig_width
    grid_height = int(width * aspect_ratio)
    
    small_image = cv2.resize(image, (width, grid_height))
    
    gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    
    indices = (gray_image / 255 * (len(ASCII_CHARS) - 1)).astype(int)
    
    ascii_rows = []
    for row in indices:
        ascii_rows.append("".join([ASCII_CHARS[i] for i in row]))
        
    return ascii_rows, small_image, grid_height

def render_ascii_to_image(ascii_rows, color_map, out_width, out_height, color_mode):
    canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    
    cell_w = out_width // len(ascii_rows[0])
    cell_h = out_height // len(ascii_rows)
    
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 0.8
    thickness = 1
    
    for r, row_text in enumerate(ascii_rows):
        for c, char in enumerate(row_text):
            x = c * cell_w
            y = (r + 1) * cell_h 
            
            if color_mode:
                color = tuple(map(int, color_map[r, c]))
            else:
                color = (0, 255, 0) 
            
            cv2.putText(canvas, char, (x, y), font, font_scale, color, thickness)
            
    return canvas

def main():
    global ENABLE_COLOR
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    
    print(f"Input Camera Source: {cam_width}x{cam_height} @ {fps} FPS")
    print(f"ASCII Grid Size: {GRID_WIDTH} cols")
    print("Press 'c' to toggle color, 'q' to quit.")

    try:
        with pyvirtualcam.Camera(width=cam_width, height=cam_height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as virt_cam:
            print(f"Virtual Camera started: {virt_cam.device}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                ascii_text, color_map, grid_h = pixel_to_ascii(frame, width=GRID_WIDTH)
                
                output_frame = render_ascii_to_image(ascii_text, color_map, cam_width, cam_height, ENABLE_COLOR)
                
                virt_cam.send(output_frame)
                
                cv2.imshow('ASCII Preview (Press q to quit)', output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    ENABLE_COLOR = not ENABLE_COLOR
                    print(f"Color Mode: {'ON' if ENABLE_COLOR else 'OFF'}")
                    
                virt_cam.sleep_until_next_frame()

    except RuntimeError as e:
        print(f"Error: {e}")
        print("Did you install OBS or a virtual camera driver?")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()