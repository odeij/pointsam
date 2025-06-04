import time
import torch
import argparse
import cv2 as cv
import numpy as np
try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    print("pyrealsense2 is not installed. Please install it using the following command: pip install pyrealsense2")
    rs = None
from utils import *

def parse_args(argv=None) -> None:
    parser = argparse.ArgumentParser(description='pointSAM')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='use CUDA for SAM. (Set to False to force CPU)')
    parser.add_argument('--img_path', default='./images/test/rgb063.png', type=str,
                        help='path to an image for segmentation.')
    parser.add_argument('--use_camera', default=False, type=bool,
                        help='use a camera [Realsense d435 is used].')
    parser.add_argument('--cad_path', default='./objects/obj_05.ply', type=str,
                        help='path to an object cad model.')
    parser.add_argument('--checkpoint', default='./weights/sam_vit_l_0b3195.pth', type=str,
                        help='path to a checkpoint.')
    parser.add_argument('--model_type', default='vit_l', type=str,
                        help='model_type for SAM corresponding to the checkpoint.')
    parser.add_argument('--save_imgs', default='./result', type=str,
                        help='path to save output images.')
    global args
    args = parser.parse_args(argv)

def overlay_mask(img, mask, color=(0, 0, 255), alpha=0.1):
    """
    Helper function to overlay a single mask on the image.
    """
    # Create a copy of the image to avoid modifying the original
    result = img.copy()
    
    # Create a colored mask
    colored_mask = np.zeros_like(result)
    colored_mask[mask == 1] = color
    
    # Overlay mask on image with transparency
    cv.addWeighted(colored_mask, alpha, result, 1 - alpha, 0, result)
    
    # Add contour for better visibility
    mask_image = mask.astype(np.uint8) * 255
    contours, _ = cv.findContours(mask_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(result, contours, -1, color, 2)
    
    return result

def update_display():
    """
    Update the display with the current state of masks and selections.
    """
    global current_display, original_img, all_masks, region_clicks, region_labels, current_region_idx
    
    # Define colors for different regions (BGR format)
    region_colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (0, 0, 128)     # Dark Red
    ]
    
    # Start with the original image
    temp_display = original_img.copy()
    
    # Draw status text
    mode_text = "MULTI REGION MODE" if multi_region_mode else f"REFINE MODE - Region {current_region_idx + 1}"
    cv.putText(temp_display, mode_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv.putText(temp_display, mode_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Display total region count
    region_count_text = f"Regions: {len(all_masks)} | Current: {current_region_idx + 1}"
    cv.putText(temp_display, region_count_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv.putText(temp_display, region_count_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Overlay all masks
    for i, mask in enumerate(all_masks):
        color = region_colors[i % len(region_colors)]
        # Make current region more prominent
        alpha = 0.7 if i == current_region_idx else 0.3
        temp_display = overlay_mask(temp_display, mask, color, alpha)
    
    # Draw all clicks for the current region
    if current_region_idx < len(region_clicks):
        for c_idx, (cx, cy) in enumerate(region_clicks[current_region_idx]):
            color = (0, 255, 0) if region_labels[current_region_idx][c_idx] == 1 else (0, 0, 255)
            cv.circle(temp_display, (cx, cy), 4, color, -1)
    
    current_display = temp_display
    cv.imshow('Segment Anything', current_display)

def click_event(event, x, y, flags, params):
    """
    Mouse click handler that handles multiple segmentation modes.
    """
    global mask_predictor, current_display, all_masks, current_region_idx
    global region_clicks, region_labels, multi_region_mode
    
    if event == cv.EVENT_LBUTTONDOWN:  # Foreground point (positive)
        if multi_region_mode:
            # In multi-region mode, create new region or select existing
            mask_clicked = False
            
            # Check if clicked on an existing mask
            for idx, mask in enumerate(all_masks):
                if mask[y, x] == 1:  # If clicked on existing mask
                    current_region_idx = idx
                    mask_clicked = True
                    print(f"Selected region {current_region_idx + 1}")
                    update_display()
                    break
            
            # If didn't click on existing mask, create new region
            if not mask_clicked:
                current_region_idx = len(all_masks)
                region_clicks.append([[x, y]])
                region_labels.append([1])
                
                # Predict mask for new region
                masks, _, _ = mask_predictor.predict(
                    point_coords=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    multimask_output=False
                )
                
                # Add new mask to collection
                all_masks.append(masks[0])
                print(f"Created new region {current_region_idx + 1}")
                update_display()
        
        else:  # Single region refinement mode
            # Add to current region's click history
            if current_region_idx < len(region_clicks):
                region_clicks[current_region_idx].append([x, y])
                region_labels[current_region_idx].append(1)
            else:
                # Should not happen, but just in case
                region_clicks.append([[x, y]])
                region_labels.append([1])
                current_region_idx = len(region_clicks) - 1
            
            # Predict with all accumulated clicks for current region
            clicks = np.array(region_clicks[current_region_idx])
            labels = np.array(region_labels[current_region_idx])
            
            masks, _, _ = mask_predictor.predict(
                point_coords=clicks,
                point_labels=labels,
                multimask_output=False
            )
            
            # Update the mask for current region
            if current_region_idx < len(all_masks):
                all_masks[current_region_idx] = masks[0]
            else:
                all_masks.append(masks[0])
            
            update_display()
    
    elif event == cv.EVENT_RBUTTONDOWN:  # Background point (negative)
        if not multi_region_mode and current_region_idx < len(region_clicks):
            # Add negative click to current region
            region_clicks[current_region_idx].append([x, y])
            region_labels[current_region_idx].append(0)
            
            # Predict with all accumulated clicks for current region
            clicks = np.array(region_clicks[current_region_idx])
            labels = np.array(region_labels[current_region_idx])
            
            masks, _, _ = mask_predictor.predict(
                point_coords=clicks,
                point_labels=labels,
                multimask_output=False
            )
            
            # Update the mask for current region
            all_masks[current_region_idx] = masks[0]
            update_display()

def select_next_region():
    """Navigate to the next region"""
    global current_region_idx, all_masks
    if all_masks:
        current_region_idx = (current_region_idx + 1) % len(all_masks)
        print(f"Selected region {current_region_idx + 1} of {len(all_masks)}")
        update_display()

def select_previous_region():
    """Navigate to the previous region"""
    global current_region_idx, all_masks
    if all_masks:
        current_region_idx = (current_region_idx - 1) % len(all_masks)
        print(f"Selected region {current_region_idx + 1} of {len(all_masks)}")
        update_display()

def d435_initializer(index: int = 0):
    if rs is None:
        print("RealSense library is not installed. Skipping Realsense camera initialization...")
        img = cv.imread(args.img_path, 1)
        points = None
        return img, points
    
    pipeline = rs.pipeline()
    config = rs.config()
    colorizer = rs.colorizer()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    depth_sensor = device.first_depth_sensor()
    desired_preset_index = index  # Example index, replace with your choice
    depth_sensor.set_option(rs.option.visual_preset, desired_preset_index)
    rs_witdh, rs_heigt = 1280, 720
    config.enable_stream(rs.stream.depth, rs_witdh, rs_heigt, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs_witdh, rs_heigt, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    pc = rs.pointcloud()
    time.sleep(1)
    try:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = cv.normalize(color_image, None, 0, 255, cv.NORM_MINMAX)
        
        # Preview in a temporary window
        preview_img = cv.addWeighted(depth_colormap, 0.5, color_image, 0.7, 0)
        cv.imshow('Preview', preview_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        points = pc.calculate(depth_frame)
        pc.map_to(depth_frame)
    finally:
        pipeline.stop()
    return color_image, points

def segment_anything(args, img):
    """
    Main function that handles the interactive segmentation process with support for
    multiple separate regions and region navigation.
    """
    global mask_predictor, original_img, current_display
    global all_masks, current_region_idx, region_clicks, region_labels, multi_region_mode
    
    # Initialize the model
    device = 'cpu'
    if args.cuda and torch.cuda.is_available():
        device = 'cuda'
    
    mask_predictor = sam_loader(checkpoint=args.checkpoint, model_type=args.model_type, device=device)
    
    # Store the original image and set as current display
    original_img = img.copy()
    current_display = original_img.copy()
    
    # Initialize segmentation variables
    all_masks = []
    region_clicks = []
    region_labels = []
    current_region_idx = 0
    multi_region_mode = True  # Start in multi-region mode
    
    # Set the image in the predictor
    mask_predictor.set_image(original_img)
    
    # Create window and set mouse callback
    cv.namedWindow('Segment Anything')
    cv.setMouseCallback('Segment Anything', click_event)
    
    # Show initial image
    cv.imshow('Segment Anything', current_display)
    
    print("=== Segment Anything Controls ===")
    print("Mode: Multi-Region (M to toggle)")
    print("- Left click: Create new region or select existing")
    print("- Right click: Add background point (in refine mode)")
    print("- 'n': Select next region")
    print("- 'p': Select previous region")
    print("- 'm': Toggle between multi-region and refinement modes")
    print("- 'r': Reset all clicks and regions")
    print("- 'c': Clear last click of current region")
    print("- 'd': Delete current region")
    print("- 's': Save all masks")
    print("- 'q': Quit")
    
    # Display help overlay
    help_display = original_img.copy()
    help_text = [
        "Controls:",
        "N/P: Next/Previous Region",
        "M: Toggle Mode",
        "R: Reset All",
        "C: Clear Last Click",
        "D: Delete Region",
        "S: Save Masks",
        "Q: Quit"
    ]
    
    y_pos = 30
    for text in help_text:
        cv.putText(help_display, text, (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv.putText(help_display, text, (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
    
    cv.imshow('Segment Anything', help_display)
    cv.waitKey(2000)  # Show help for 2 seconds
    
    # Update display to clear help
    update_display()
    
    # Main interaction loop
    while True:
        key = cv.waitKey(100) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('m'):  # Toggle mode
            multi_region_mode = not multi_region_mode
            mode_str = "Multi-Region" if multi_region_mode else f"Refining Region {current_region_idx + 1}"
            print(f"Mode: {mode_str}")
            update_display()
        elif key == ord('n'):  # Next region
            select_next_region()
        elif key == ord('p'):  # Previous region
            select_previous_region()
        elif key == ord('r'):  # Reset all
            all_masks = []
            region_clicks = []
            region_labels = []
            current_region_idx = 0
            current_display = original_img.copy()
            cv.imshow('Segment Anything', current_display)
            print("Reset all regions")
        elif key == ord('c') and not multi_region_mode:  # Clear last click of current region
            if (current_region_idx < len(region_clicks) and 
                len(region_clicks[current_region_idx]) > 0):
                
                region_clicks[current_region_idx].pop()
                region_labels[current_region_idx].pop()
                
                # If region still has clicks, update mask
                if len(region_clicks[current_region_idx]) > 0:
                    clicks = np.array(region_clicks[current_region_idx])
                    labels = np.array(region_labels[current_region_idx])
                    
                    masks, _, _ = mask_predictor.predict(
                        point_coords=clicks,
                        point_labels=labels,
                        multimask_output=False
                    )
                    
                    all_masks[current_region_idx] = masks[0]
                else:
                    # If no clicks left, remove the mask
                    all_masks[current_region_idx] = np.zeros_like(all_masks[current_region_idx])
                
                update_display()
                print("Removed last click from current region")
        elif key == ord('d'):  # Delete current region
            if current_region_idx < len(all_masks):
                all_masks.pop(current_region_idx)
                region_clicks.pop(current_region_idx)
                region_labels.pop(current_region_idx)
                
                if len(all_masks) == 0:
                    current_region_idx = 0
                    current_display = original_img.copy()
                else:
                    current_region_idx = min(current_region_idx, len(all_masks) - 1)
                    update_display()
                
                cv.imshow('Segment Anything', current_display)
                print(f"Deleted region. {len(all_masks)} regions remaining.")
        elif key == ord('s'):  # Save
            if args.save_imgs and all_masks:
                # Create directories if they don't exist
                import os
                os.makedirs(args.save_imgs, exist_ok=True)
                
                # Create combined mask for visualization
                combined_mask = np.zeros_like(original_img[:,:,0], dtype=np.uint8)
                for i, mask in enumerate(all_masks):
                    # Add region index to the mask (1-based)
                    combined_mask[mask == 1] = i + 1
                
                # Save individual masks
                for i, mask in enumerate(all_masks):
                    cv.imwrite(f'{args.save_imgs}/mask_region_{i+1}.png', mask * 255)
                
                # Save combined mask (labeled)
                cv.imwrite(f'{args.save_imgs}/mask_combined.png', combined_mask * 50)  # Scale for visibility
                
                # Save segmented image
                cv.imwrite(f'{args.save_imgs}/segmented.png', current_display)
                
                # Save region data for potential reuse
                for i, (clicks, labels) in enumerate(zip(region_clicks, region_labels)):
                    np.save(f'{args.save_imgs}/region_{i+1}_clicks.npy', np.array(clicks))
                    np.save(f'{args.save_imgs}/region_{i+1}_labels.npy', np.array(labels))
                
                print(f"Saved {len(all_masks)} masks to {args.save_imgs}")
                
                # Show a brief confirmation overlay
                temp_display = current_display.copy()
                cv.putText(temp_display, f"Saved {len(all_masks)} masks!", 
                          (img.shape[1]//2 - 100, img.shape[0]//2), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                cv.putText(temp_display, f"Saved {len(all_masks)} masks!", 
                          (img.shape[1]//2 - 100, img.shape[0]//2), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                cv.imshow('Segment Anything', temp_display)
                cv.waitKey(1000)  # Show confirmation for 1 second
                update_display()
    
    cv.destroyAllWindows()
    
    # Return the combined mask
    if all_masks:
        combined_mask = np.zeros_like(all_masks[0], dtype=np.uint8)
        for i, mask in enumerate(all_masks):
            combined_mask[mask == 1] = i + 1
        return combined_mask
    else:
        return None

if __name__ == "__main__":
    parse_args()
    
    if args.use_camera:
        print("Initialize Realsense d435...")
        img, points = d435_initializer()
    else:
        assert args.img_path is not None, "image_path is not provided!!!"
        img = cv.imread(args.img_path, 1)
        points = None
    
    mask = segment_anything(args=args, img=img)
    
    # Optional: generate point cloud if needed
    # if points is not None and mask is not None:
    #     pts = gen_pointcloud(points=points, segment_img=mask)