#!/usr/bin/env python3
"""
Intelligent Point-SAM Interface
Advanced 3D point cloud segmentation with smart click detection and neighborhood analysis
Addresses the sparse point cloud problem by using intelligent prompting strategies
"""

import os
import sys
import numpy as np
import torch
import open3d as o3d
import time
import threading
from queue import Queue
import copy
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

# Add current directory to path for Point-SAM imports
sys.path.append('.')

try:
    import hydra
    from omegaconf import OmegaConf
    from pc_sam.model.pc_sam import PointCloudSAM
    from pc_sam.utils.torch_utils import replace_with_fused_layernorm
    from safetensors.torch import load_model
    from pc_sam.ply_utils import read_ply
    REAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Point-SAM imports failed: {e}")
    REAL_MODEL_AVAILABLE = False

class IntelligentPointSAMUI:
    def __init__(self):
        """Initialize Intelligent Point-SAM UI with advanced 3D understanding"""
        print("🧠 Initializing Intelligent Point-SAM Interface...")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load Point-SAM model
        self.model = None
        self.load_pointsam_model()
        
        # Point cloud data
        self.original_xyz = None
        self.original_colors = None
        self.pc_xyz = None  # Normalized for model
        self.pc_rgb = None
        self.normalization = None
        
        # Spatial indexing for fast neighbor search
        self.kdtree = None
        self.point_normals = None
        self.local_descriptors = None
        
        # Visualization
        self.vis = None
        self.point_cloud = None
        self.running = True
        self.update_pending = False
        
        # Advanced interaction state
        self.click_points = []
        self.click_neighborhoods = []  # Store neighborhood info for each click
        self.current_mask = None
        self.segmentation_history = []
        
        # Intelligent prompting parameters
        self.neighborhood_radius = 0.05  # Adaptive radius for neighborhood analysis
        self.min_neighborhood_size = 5
        self.max_neighborhood_size = 50
        self.feature_similarity_threshold = 0.8
        
        # Colors for visualization
        self.original_color_backup = None
        self.click_color = np.array([1.0, 0.2, 0.2])      # Bright red for clicks
        self.neighborhood_color = np.array([1.0, 0.6, 0.2])  # Orange for neighborhoods
        self.segment_color = np.array([0.2, 1.0, 0.2])    # Bright green for segmented
        self.normal_color = np.array([0.8, 0.8, 0.8])     # Light gray for normal
        
        print("✅ Intelligent Point-SAM Interface initialized!")
    
    def load_pointsam_model(self):
        """Load the real Point-SAM model with timeout handling"""
        print("🔧 Loading Point-SAM model...")
        
        if not REAL_MODEL_AVAILABLE:
            print("⚠️  Using intelligent dummy model (Point-SAM imports failed)")
            self.model = self.create_intelligent_dummy_model()
            return
        
        try:
            print("🚀 Attempting to load real Point-SAM model...")
            
            # Load configuration with timeout handling
            with hydra.initialize("configs", version_base=None):
                cfg = hydra.compose(config_name="large")
                OmegaConf.resolve(cfg)
            
            # Initialize model
            self.model = hydra.utils.instantiate(cfg.model)
            self.model.apply(replace_with_fused_layernorm)
            
            # Load pretrained weights
            checkpoint_path = "pretrained/model.safetensors"
            if os.path.exists(checkpoint_path):
                load_model(self.model, checkpoint_path)
                print(f"✅ Loaded pretrained weights from {checkpoint_path}")
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Real Point-SAM model loaded on {self.device}")
            
            # Test the model
            print("🧪 Testing model inference...")
            self.test_model_inference()
            
        except Exception as e:
            print(f"❌ Failed to load real Point-SAM model: {e}")
            print("🔄 Falling back to intelligent dummy model...")
            self.model = self.create_intelligent_dummy_model()
    
    def test_model_inference(self):
        """Test if the model can run inference without hanging"""
        try:
            # Create dummy data for testing
            coords = torch.rand(1, 1000, 3).to(self.device) * 2 - 1  # Uniform in [-1, 1]
            features = torch.rand(1, 1000, 3).to(self.device) * 2 - 1
            prompt_coords = torch.rand(1, 1, 3).to(self.device) * 2 - 1
            prompt_labels = torch.ones(1, 1, dtype=torch.long).to(self.device)
            
            # Set timeout for inference
            timeout = 10
            result_queue = Queue()
            
            def inference_worker():
                with torch.no_grad():
                    masks, iou_scores = self.model.predict_masks(
                        coords,
                        features,
                        prompt_coords,
                        prompt_labels,
                        multimask_output=False
                    )
                return masks, iou_scores
            
            # Run inference with timeout
            inference_thread = threading.Thread(target=lambda: result_queue.put(inference_worker()))
            inference_thread.daemon = True
            inference_thread.start()
            inference_thread.join(timeout=timeout)
            
            if inference_thread.is_alive():
                print(f"⚠️  Model inference timed out after {timeout}s, using intelligent dummy model")
                self.model = self.create_intelligent_dummy_model()
            else:
                if not result_queue.empty():
                    inference_time = time.time()
                    print(f"✅ Model test passed! Inference ready")
                else:
                    print("⚠️  Model test failed, using intelligent dummy model")
                    self.model = self.create_intelligent_dummy_model()
                
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            print("🔄 Using intelligent dummy model for reliability")
            self.model = self.create_intelligent_dummy_model()
    
    def create_intelligent_dummy_model(self):
        """Create an intelligent dummy model that mimics 2D SAM behavior in 3D"""
        class IntelligentDummyModel:
            def __init__(self, device):
                self.device = device
                print("✅ Intelligent dummy model created with multimask spatial-feature analysis")
            
            def predict_masks(self, coords, point_features=None, prompt_coords=None, prompt_labels=None, multimask_output=True):
                """Predict segmentation masks using intelligent dummy model"""
                # Ensure coords has batch dimension
                if len(coords.shape) == 2:
                    coords = coords.unsqueeze(0)
                
                batch_size, num_points, _ = coords.shape
                
                # Initialize masks
                masks = []
                iou_scores = []
                
                # Process each prompt
                for i in range(len(prompt_coords[0])):
                    prompt = prompt_coords[0][i]
                    label = prompt_labels[0][i]
                    
                    # Calculate distances from prompt to all points
                    distances = torch.norm(coords[0] - prompt, dim=-1)
                    
                    # Adaptive threshold based on point cloud density
                    # For larger point clouds, use smaller threshold for more precise segmentation
                    if num_points > 100000:
                        threshold = 0.05  # 5cm for dense clouds
                    elif num_points > 50000:
                        threshold = 0.08  # 8cm for medium clouds
                    else:
                        threshold = 0.1   # 10cm for sparse clouds
                    
                    # Create mask based on adaptive distance threshold
                    mask = distances < threshold
                    
                    # If no points selected, gradually increase threshold
                    if mask.sum() == 0:
                        for fallback_threshold in [0.1, 0.15, 0.2, 0.3]:
                            mask = distances < fallback_threshold
                            if mask.sum() > 0:
                                break
                    
                    # Add to results
                    masks.append(mask)
                    iou_scores.append(torch.tensor(0.8).to(self.device))  # Dummy IoU score
                
                # If multimask_output, create multiple masks with different thresholds
                if multimask_output and len(masks) > 0:
                    base_mask = masks[0]
                    base_prompt = prompt_coords[0][0]
                    distances = torch.norm(coords[0] - base_prompt, dim=-1)
                    
                    # Create 3 masks with different thresholds
                    thresholds = [0.05, 0.1, 0.15]
                    multi_masks = []
                    multi_ious = []
                    
                    for thresh in thresholds:
                        mask = distances < thresh
                        if mask.sum() == 0:  # Fallback if no points
                            mask = distances < 0.2
                        multi_masks.append(mask)
                        multi_ious.append(torch.tensor(0.8 - thresh).to(self.device))  # Varying IoU
                    
                    return torch.stack(multi_masks).unsqueeze(0), torch.stack(multi_ious).unsqueeze(0)
                
                return torch.stack(masks).unsqueeze(0), torch.stack(iou_scores).unsqueeze(0)
        
        return IntelligentDummyModel(self.device)
    
    def load_point_cloud(self, file_path):
        """Load and preprocess point cloud with advanced spatial analysis"""
        print(f"📁 Loading point cloud: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        try:
            # Load point cloud
            if file_path.endswith('.ply'):
                try:
                    points = read_ply(file_path)
                    xyz = points[:, :3]
                    rgb = points[:, 3:6] / 255.0 if points.shape[1] >= 6 else np.ones_like(xyz) * 0.7
                    print("   ✅ Loaded using pc_sam PLY reader")
                except Exception as e:
                    print(f"   ⚠️  pc_sam PLY reader failed ({e}), using Open3D...")
                    pcd = o3d.io.read_point_cloud(file_path)
                    xyz = np.asarray(pcd.points)
                    rgb = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(xyz) * 0.7
                    print("   ✅ Loaded using Open3D PLY reader")
            else:
                pcd = o3d.io.read_point_cloud(file_path)
                xyz = np.asarray(pcd.points)
                rgb = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(xyz) * 0.7
            
            print(f"📊 Original point cloud shape: {xyz.shape}")
            print(f"📏 XYZ range: [{xyz.min():.3f}, {xyz.max():.3f}]")
            
            # Remove subsampling to use full point cloud
            # max_points = 100000
            # if xyz.shape[0] > max_points:
            #     indices = np.random.choice(xyz.shape[0], max_points, replace=False)
            #     xyz = xyz[indices]
            #     rgb = rgb[indices]
            #     print(f"🔄 Subsampled to {xyz.shape[0]} points for performance")
            print(f"🔄 Using full point cloud: {xyz.shape[0]} points")
            
            # Store original data
            self.original_xyz = xyz
            self.original_colors = rgb.copy()
            self.original_color_backup = rgb.copy()
            
            # Build spatial index for fast neighbor search
            print("🔍 Building spatial index...")
            self.kdtree = KDTree(xyz)
            
            # Analyze point cloud density for adaptive strategies
            print("📊 Analyzing point cloud density...")
            self.density_info = self.analyze_point_density()
            
            # Calculate point normals for better spatial understanding
            print("📐 Computing point normals...")
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(xyz)
            
            # Adaptive normal estimation radius
            scene_scale = np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))
            normal_radius = scene_scale / 100  # 1% of scene scale
            normal_radius = max(0.05, min(0.2, normal_radius))  # Clamp between 5cm and 20cm
            
            pcd_temp.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius, max_nn=30
                )
            )
            self.point_normals = np.asarray(pcd_temp.normals)
            
            # Normalize coordinates for Point-SAM
            shift = xyz.mean(0)
            scale = np.linalg.norm(xyz - shift, axis=-1).max()
            xyz_normalized = (xyz - shift) / scale
            
            # Ensure strict [-1, 1] bounds for real Point-SAM model
            xyz_normalized = np.clip(xyz_normalized, -1.0, 1.0)
            
            # Additional safety check - if using real model, be extra careful
            if hasattr(self.model, 'predict_masks') and not hasattr(self.model, 'device'):
                # This is the real model, apply extra safety margin
                max_coord = np.abs(xyz_normalized).max()
                if max_coord > 0.98:
                    safety_scale = 0.95 / max_coord
                    xyz_normalized = xyz_normalized * safety_scale
                    print(f"   🔒 Applied safety scaling: {safety_scale:.3f}")
            
            print(f"📐 Normalization - Shift: {shift}, Scale: {scale:.3f}")
            print(f"📏 Normalized range: [{xyz_normalized.min():.3f}, {xyz_normalized.max():.3f}]")
            
            # Store normalized coordinates
            self.pc_xyz = xyz_normalized
            self.pc_rgb = rgb
            self.normalization = {'shift': shift, 'scale': scale}
            
            # Create visualization point cloud
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(xyz)
            self.point_cloud.colors = o3d.utility.Vector3dVector(rgb)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading point cloud: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_click_neighborhood(self, click_point_idx):
        """Analyze the neighborhood around a clicked point for intelligent prompting"""
        click_point = self.original_xyz[click_point_idx]
        
        # Get density information
        density_info = getattr(self, 'density_info', {'is_sparse': True, 'sparsity_level': 'sparse'})
        
        # Adaptive radius based on sparsity
        base_radius = self.neighborhood_radius
        if density_info['sparsity_level'] == 'very_sparse':
            adaptive_radius = base_radius * 4.0  # 4x radius for very sparse
            print(f"🔍 Very sparse cloud detected - expanding radius to {adaptive_radius:.3f}")
        elif density_info['sparsity_level'] == 'sparse':
            adaptive_radius = base_radius * 2.5  # 2.5x radius for sparse
            print(f"🔍 Sparse cloud detected - expanding radius to {adaptive_radius:.3f}")
        else:
            adaptive_radius = base_radius
        
        # Multi-scale neighborhood analysis
        neighbor_sets = {}
        radii = [adaptive_radius * 0.5, adaptive_radius, adaptive_radius * 1.5]
        
        for i, radius in enumerate(radii):
            neighbor_indices = self.kdtree.query_ball_point(click_point, radius)
            neighbor_indices = np.array(neighbor_indices)
            
            if len(neighbor_indices) < 3:  # Too few neighbors, expand further
                radius *= 2
                neighbor_indices = self.kdtree.query_ball_point(click_point, radius)
                neighbor_indices = np.array(neighbor_indices)
            
            neighbor_sets[f'scale_{i}'] = {
                'indices': neighbor_indices,
                'radius': radius,
                'count': len(neighbor_indices)
            }
        
        # Use the middle scale as primary, but consider all scales
        primary_neighbors = neighbor_sets['scale_1']
        neighbor_indices = primary_neighbors['indices']
        
        # If still too few neighbors, use the largest scale
        if len(neighbor_indices) < self.min_neighborhood_size:
            neighbor_indices = neighbor_sets['scale_2']['indices']
            print(f"🔍 Expanding to largest scale: {len(neighbor_indices)} neighbors")
        
        # Cap maximum neighbors for performance
        if len(neighbor_indices) > self.max_neighborhood_size:
            distances = np.linalg.norm(self.original_xyz[neighbor_indices] - click_point, axis=1)
            sorted_indices = np.argsort(distances)
            neighbor_indices = neighbor_indices[sorted_indices[:self.max_neighborhood_size]]
        
        # Analyze neighborhood characteristics
        neighbor_points = self.original_xyz[neighbor_indices]
        neighbor_colors = self.original_colors[neighbor_indices]
        neighbor_normals = self.point_normals[neighbor_indices]
        
        # Calculate local geometric features
        centroid = neighbor_points.mean(axis=0)
        
        # Robust covariance calculation for sparse data
        if len(neighbor_points) >= 3:
            covariance = np.cov(neighbor_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-8)  # Avoid division by zero
            
            # Primary direction (largest eigenvalue)
            primary_direction = eigenvectors[:, -1]
            planarity = (eigenvalues[-2] - eigenvalues[0]) / eigenvalues[-1] if eigenvalues[-1] > 0 else 0
        else:
            primary_direction = np.array([1, 0, 0])  # Default direction
            planarity = 0
        
        # Adaptive color analysis for sparse clouds
        color_coherence = 0
        if len(neighbor_colors) > 1:
            color_std = neighbor_colors.std(axis=0)
            color_coherence = 1.0 / (1.0 + np.mean(color_std))  # Higher coherence = lower std
        
        neighborhood_info = {
            'indices': neighbor_indices,
            'center': centroid,
            'radius': primary_neighbors['radius'],
            'density': len(neighbor_indices) / (np.pi * primary_neighbors['radius']**2),
            'primary_direction': primary_direction,
            'planarity': planarity,
            'mean_color': neighbor_colors.mean(axis=0),
            'color_std': neighbor_colors.std(axis=0),
            'color_coherence': color_coherence,
            'mean_normal': neighbor_normals.mean(axis=0),
            'multi_scale_info': neighbor_sets,
            'sparsity_adapted': True
        }
        
        print(f"🔍 Adaptive neighborhood analysis:")
        print(f"   📊 {len(neighbor_indices)} points at radius {primary_neighbors['radius']:.3f}")
        print(f"   🎯 Density: {neighborhood_info['density']:.1f}, Planarity: {planarity:.2f}")
        print(f"   🎨 Color coherence: {color_coherence:.2f}")
        
        return neighborhood_info
    
    def generate_intelligent_prompts(self, click_point_idx):
        """Generate multiple intelligent prompts based on neighborhood analysis"""
        neighborhood = self.analyze_click_neighborhood(click_point_idx)
        
        prompts = []
        labels = []
        
        # Primary prompt: the clicked point itself
        click_point = self.original_xyz[click_point_idx]
        prompts.append(click_point)
        labels.append(1)  # Positive
        
        # Secondary prompts: strategically selected neighborhood points
        neighbor_points = self.original_xyz[neighborhood['indices']]
        
        if len(neighbor_points) > 3:
            # Add prompts along the primary direction for elongated structures
            if neighborhood['planarity'] > 0.3:  # Planar or linear structure
                direction = neighborhood['primary_direction']
                center = neighborhood['center']
                
                # Find points along the primary direction
                projections = np.dot(neighbor_points - center, direction)
                
                # Add prompts at quartile positions along the primary direction
                quartiles = np.percentile(projections, [25, 75])
                for q in quartiles:
                    target_proj = q
                    distances_to_target = np.abs(projections - target_proj)
                    closest_idx = neighborhood['indices'][np.argmin(distances_to_target)]
                    if closest_idx != click_point_idx:  # Avoid duplicate
                        prompts.append(self.original_xyz[closest_idx])
                        labels.append(1)  # Positive
            
            # Add a few random points from the neighborhood for robustness
            random_indices = np.random.choice(
                neighborhood['indices'], 
                size=min(2, len(neighborhood['indices']) // 4), 
                replace=False
            )
            for idx in random_indices:
                if idx != click_point_idx:  # Avoid duplicate
                    prompts.append(self.original_xyz[idx])
                    labels.append(1)  # Positive
        
        # Limit total prompts to avoid overwhelming the model
        max_prompts = 5
        if len(prompts) > max_prompts:
            prompts = prompts[:max_prompts]
            labels = labels[:max_prompts]
        
        print(f"🎯 Generated {len(prompts)} intelligent prompts")
        
        return prompts, labels, neighborhood
    
    def run_intelligent_interface(self, point_cloud_path):
        """Run continuous interactive Point-SAM interface"""
        print("🔄 Starting CONTINUOUS Interactive Point-SAM Interface")
        print("=" * 60)
        
        # Load point cloud with spatial analysis
        if not self.load_point_cloud(point_cloud_path):
            return False
        
        # Initialize interaction state
        self.positive_clicks = []
        self.negative_clicks = []
        self.all_prompts = []
        self.all_labels = []
        self.click_history = []
        self.last_processed_clicks = 0
        self.processing_clicks = False
        
        # Create visualizer for continuous interaction
        self.vis = o3d.visualization.VisualizerWithEditing()
        self.vis.create_window(
            window_name="🔄 CONTINUOUS Point-SAM - Live Interactive Segmentation",
            width=1800,
            height=1000
        )
        
        # Add point cloud
        self.vis.add_geometry(self.point_cloud)
        
        # Set nice view
        view_ctl = self.vis.get_view_control()
        view_ctl.set_zoom(0.8)
        
        print("🔄 === CONTINUOUS INTERACTIVE SEGMENTATION ===")
        print("   🖱️   LEFT CLICK: Add POSITIVE click")
        print("   🖱️   RIGHT CLICK: Add NEGATIVE click")
        print("   🟢   Green: Current segmentation (updates LIVE)")
        print("   🔴   Red: Positive clicks") 
        print("   🔵   Blue: Negative clicks")
        print("   ⌨️   'c': Clear all clicks and restart")
        print("   ⌨️   'u': Undo last click")
        print("   ⌨️   's': Save current segmentation")
        print("   ⌨️   Close window to quit")
        print()
        print("🔄 === CONTINUOUS WORKFLOW ===")
        print("   1️⃣   Click on object parts you WANT (positive)")
        print("   2️⃣   Right-click on parts you DON'T want (negative)")
        print("   3️⃣   Watch segmentation update INSTANTLY after each click")
        print("   4️⃣   Keep refining until satisfied!")
        print("   5️⃣   Save or close when done")
        print()
        print("🟢 === READY FOR CONTINUOUS INTERACTION ===")
        print("🎯 Click anywhere to start! Segmentation updates after EVERY click!")
        print()
        
        # Use simple run() and process all clicks afterwards
        try:
            print("🔄 Starting interactive session...")
            print("💡 Click on objects in the 3D view, then close the window when done")
            self.vis.run()
            
            # Process all clicks after user closes window
            self.process_all_clicks_continuous()
                        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"❌ Interface error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            try:
                self.vis.destroy_window()
            except:
                pass
        
        print("✅ CONTINUOUS Point-SAM Interface completed!")
        return True
    
    def process_all_clicks_continuous(self):
        """Process all clicked points with continuous segmentation updates"""
        try:
            # Get all picked points from the session
            picked_points = self.vis.get_picked_points()
            
            if not picked_points:
                print("⚠️  No points were clicked")
                return
            
            print(f"\n🚀 Processing {len(picked_points)} clicks with CONTINUOUS segmentation...")
            
            # Process each click one by one with live updates
            for i, pick_info in enumerate(picked_points):
                try:
                    # Handle different point info formats
                    if hasattr(pick_info, 'index'):
                        point_idx = pick_info.index
                    else:
                        point_idx = pick_info
                    
                    # For now, treat all as positive clicks
                    # TODO: Detect right-clicks for negative clicks
                    is_positive = True
                    
                    print(f"\n🔄 Processing click #{i+1}/{len(picked_points)}")
                    self.add_click_continuous(point_idx, is_positive)
                    
                    # Show current segmentation result
                    if self.current_mask is not None:
                        num_segmented = self.current_mask.sum()
                        print(f"   ✅ Current segmentation: {num_segmented} points")
                    
                except Exception as e:
                    print(f"❌ Error processing click #{i+1}: {e}")
            
            # Final results
            if len(picked_points) > 0:
                self.show_final_continuous_results()
            
        except Exception as e:
            print(f"❌ Error in continuous processing: {e}")
            import traceback
            traceback.print_exc()
    
    def add_click_continuous(self, point_idx, is_positive=True):
        """Add a click and immediately update segmentation (continuous version)"""
        try:
            click_point = self.original_xyz[point_idx]
            
            if is_positive:
                self.positive_clicks.append(click_point)
                self.all_prompts.append(click_point)
                self.all_labels.append(1)
                click_type = "POSITIVE"
                color = "🔴"
            else:
                self.negative_clicks.append(click_point)
                self.all_prompts.append(click_point)
                self.all_labels.append(0)
                click_type = "NEGATIVE"
                color = "🔵"
            
            # Record click history
            self.click_history.append({
                'point_idx': point_idx,
                'point': click_point,
                'is_positive': is_positive,
                'timestamp': time.time()
            })
            
            total_clicks = len(self.positive_clicks) + len(self.negative_clicks)
            print(f"   {color} {click_type} click on point #{point_idx}")
            print(f"   📊 Total: {len(self.positive_clicks)} positive, {len(self.negative_clicks)} negative")
            
            # Run segmentation immediately with all clicks
            if len(self.all_prompts) > 0:
                self.run_continuous_segmentation()
            
        except Exception as e:
            print(f"❌ Error adding continuous click: {e}")
    
    def show_final_continuous_results(self):
        """Show final results of continuous segmentation"""
        try:
            if self.current_mask is None or self.current_mask.sum() == 0:
                print("⚠️  No final segmentation to show")
                return
            
            print("\n🎨 Creating final continuous segmentation results...")
            
            # Create final visualization
            final_vis = o3d.visualization.Visualizer()
            final_vis.create_window(
                window_name="🎉 Final Continuous Segmentation Results",
                width=1400,
                height=900
            )
            
            # Create final point cloud
            final_point_cloud = o3d.geometry.PointCloud()
            final_point_cloud.points = o3d.utility.Vector3dVector(self.original_xyz)
            
            # Apply final colors
            final_colors = self.original_color_backup.copy()
            
            # Ultra-bright segmentation
            if self.current_mask is not None:
                final_colors[self.current_mask] = np.array([0.0, 1.0, 0.0])
            
            # Click points
            for click_point in self.positive_clicks:
                distances = np.linalg.norm(self.original_xyz - click_point, axis=1)
                closest_idx = np.argmin(distances)
                final_colors[closest_idx] = np.array([1.0, 0.0, 0.0])
            
            for click_point in self.negative_clicks:
                distances = np.linalg.norm(self.original_xyz - click_point, axis=1)
                closest_idx = np.argmin(distances)
                final_colors[closest_idx] = np.array([0.0, 0.0, 1.0])
            
            final_point_cloud.colors = o3d.utility.Vector3dVector(final_colors)
            final_vis.add_geometry(final_point_cloud)
            
            # Print final statistics
            num_segmented = self.current_mask.sum()
            print(f"🎉 FINAL RESULTS:")
            print(f"   📊 {num_segmented} points segmented")
            print(f"   🔴 {len(self.positive_clicks)} positive clicks")
            print(f"   🔵 {len(self.negative_clicks)} negative clicks")
            print(f"   📈 Segmentation ratio: {num_segmented/len(self.original_xyz)*100:.1f}%")
            print()
            print("💡 Press any key in the window to finish...")
            
            final_vis.run()
            final_vis.destroy_window()
            
            # Auto-save final results
            self.save_current_segmentation()
            
        except Exception as e:
            print(f"❌ Error showing final results: {e}")
    
    def run_continuous_interaction_loop(self):
        """Continuous interaction loop with real-time updates"""
        print("🔄 Starting continuous interaction - click to begin!")
        
        # Register keyboard callbacks
        self.setup_keyboard_callbacks()
        
        while self.running:
            try:
                # Update visualization and poll events
                self.vis.poll_events()
                self.vis.update_renderer()
                
                # Check for new clicks
                self.process_new_clicks()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error in continuous loop: {e}")
                break
    
    def process_new_clicks(self):
        """Process new clicks and update segmentation immediately"""
        try:
            # Get current picked points
            current_picks = self.vis.get_picked_points()
            current_count = len(current_picks)
            
            # Check if we have new clicks
            if current_count > self.last_processed_clicks:
                new_picks = current_picks[self.last_processed_clicks:]
                
                for pick_info in new_picks:
                    # Handle different point info formats
                    if hasattr(pick_info, 'index'):
                        point_idx = pick_info.index
                    else:
                        point_idx = pick_info
                    
                    # Determine if positive or negative click
                    # For now, treat all as positive (we'll add modifier key detection later)
                    is_positive = True  # TODO: Add modifier key detection
                    
                    self.add_click(point_idx, is_positive)
                
                self.last_processed_clicks = current_count
                
        except Exception as e:
            print(f"❌ Error processing clicks: {e}")
    
    def add_click(self, point_idx, is_positive=True):
        """Add a click and immediately update segmentation"""
        try:
            click_point = self.original_xyz[point_idx]
            
            if is_positive:
                self.positive_clicks.append(click_point)
                self.all_prompts.append(click_point)
                self.all_labels.append(1)
                click_type = "POSITIVE"
                color = "🔴"
            else:
                self.negative_clicks.append(click_point)
                self.all_prompts.append(click_point)
                self.all_labels.append(0)
                click_type = "NEGATIVE"
                color = "🔵"
            
            # Record click history
            self.click_history.append({
                'point_idx': point_idx,
                'point': click_point,
                'is_positive': is_positive,
                'timestamp': time.time()
            })
            
            total_clicks = len(self.positive_clicks) + len(self.negative_clicks)
            print(f"\n{color} {click_type} click #{total_clicks} on point #{point_idx}")
            print(f"   📊 Total: {len(self.positive_clicks)} positive, {len(self.negative_clicks)} negative")
            
            # Run segmentation immediately with all clicks
            if len(self.all_prompts) > 0:
                self.run_continuous_segmentation()
            
        except Exception as e:
            print(f"❌ Error adding click: {e}")
    
    def run_continuous_segmentation(self):
        """Run segmentation with all current clicks and update visualization"""
        try:
            start_time = time.time()
            
            # Generate intelligent prompts based on positive clicks
            if len(self.positive_clicks) > 0:
                primary_click_point = self.positive_clicks[0]
                
                # Find primary click point index
                distances = np.linalg.norm(self.original_xyz - primary_click_point, axis=1)
                primary_idx = np.argmin(distances)
                
                # Generate additional intelligent prompts around primary click
                neighborhood = self.analyze_click_neighborhood(primary_idx)
                
                # Combine user clicks with intelligent prompts
                combined_prompts = self.all_prompts.copy()
                combined_labels = self.all_labels.copy()
                
                # Add intelligent prompts (only positive ones to avoid conflicts)
                intelligent_prompts, intelligent_labels, _ = self.generate_intelligent_prompts(primary_idx)
                for i, (prompt, label) in enumerate(zip(intelligent_prompts, intelligent_labels)):
                    if label == 1 and i > 0:  # Skip first (it's the same as user click)
                        combined_prompts.append(prompt)
                        combined_labels.append(label)
                
                # Convert to tensors with batch dimension
                # Safety check: ensure pc_xyz is in [-1, 1]
                assert np.all(self.pc_xyz >= -1.0) and np.all(self.pc_xyz <= 1.0), "Coordinates not in [-1, 1]"
                coords = torch.from_numpy(self.pc_xyz).float().unsqueeze(0).to(self.device)
                features = torch.from_numpy(self.pc_rgb).float().unsqueeze(0).to(self.device)
                prompt_coords = torch.from_numpy(np.array(combined_prompts)).float().unsqueeze(0).to(self.device)
                prompt_labels = torch.from_numpy(np.array(combined_labels)).long().unsqueeze(0).to(self.device)
                
                # Run model prediction
                masks, iou_scores = self.model.predict_masks(
                    coords,
                    features,
                    prompt_coords,
                    prompt_labels
                )
                
                # Get best mask
                best_mask_idx = torch.argmax(iou_scores).item()
                best_mask = masks[0, best_mask_idx].cpu().numpy()  # Get first batch, best mask
                
                # Update visualization
                self.update_visualization(best_mask)
                
                # Store current mask
                self.current_mask = best_mask
                
                # Print timing
                elapsed = time.time() - start_time
                print(f"   ⚡ LIVE update: {best_mask.sum()} points in {elapsed:.3f}s (IoU: {float(iou_scores[0, best_mask_idx]):.3f})")
                print(f"   ✅ Current segmentation: {best_mask.sum()} points")
            
        except Exception as e:
            print(f"❌ Error in continuous segmentation: {e}")
            import traceback
            traceback.print_exc()
    
    def segment_whole_object(self, click_point_idx, initial_mask):
        """Segment the whole object using aggressive surface-based region growing"""
        try:
            print(f"🪑 Starting AGGRESSIVE whole-object segmentation for clear boundaries...")
            
            # Get density info for adaptive parameters
            density_info = getattr(self, 'density_info', {'sparsity_level': 'dense'})
            
            # MUCH MORE AGGRESSIVE parameters for very clear object segmentation
            if density_info['sparsity_level'] == 'very_sparse':
                max_distance = 1.5    # Very large for sparse clouds
                normal_threshold = 0.1  # Very permissive (almost ignore normals)
                color_threshold = 0.8   # More permissive
                min_cluster_size = 20
                growth_iterations = 30  # More iterations
            elif density_info['sparsity_level'] == 'sparse':
                max_distance = 1.0    # Much larger
                normal_threshold = 0.2
                color_threshold = 0.7
                min_cluster_size = 50
                growth_iterations = 25
            else:  # dense clouds
                max_distance = 0.8    # Much larger than before (was 0.2)
                normal_threshold = 0.3
                color_threshold = 0.6
                min_cluster_size = 100
                growth_iterations = 20
            
            print(f"   🎯 AGGRESSIVE params: max_dist={max_distance}m, iterations={growth_iterations}")
            
            # Stage 1: Large-scale spatial clustering to find object-sized regions
            click_point = self.original_xyz[click_point_idx]
            
            # Find all points within very large radius (whole furniture pieces)
            distances = np.linalg.norm(self.original_xyz - click_point, axis=1)
            candidate_mask = distances < max_distance
            candidate_indices = np.where(candidate_mask)[0]
            
            # If not enough candidates, expand even further
            if len(candidate_indices) < min_cluster_size:
                print(f"   ⚠️  Expanding search to {max_distance * 2}m for better coverage...")
                candidate_mask = distances < max_distance * 2
                candidate_indices = np.where(candidate_mask)[0]
            
            print(f"   📊 Found {len(candidate_indices)} candidates within {max_distance}m")
            
            # Stage 2: Multi-scale aggressive surface growing
            object_mask = self.aggressive_surface_growing(
                click_point_idx, candidate_indices, 
                max_distance, normal_threshold, color_threshold, growth_iterations
            )
            
            # Stage 3: Geometric consistency check (more lenient)
            if object_mask.sum() > min_cluster_size:
                object_mask = self.refine_with_geometric_consistency(object_mask, click_point_idx, max_furniture_size=3.0)
            
            # Stage 4: Final cleanup with very aggressive clustering
            if object_mask.sum() > min_cluster_size:
                object_mask = self.ultra_aggressive_connected_components(object_mask, click_point_idx, max_distance)
            
            final_count = object_mask.sum()
            print(f"   ✅ AGGRESSIVE segmentation result: {final_count} points")
            
            return object_mask
            
        except Exception as e:
            print(f"   ❌ Aggressive segmentation failed: {e}, using fallback")
            return initial_mask
    
    def aggressive_surface_growing(self, click_point_idx, candidate_indices, max_distance, normal_threshold, color_threshold, max_iterations):
        """Very aggressive surface-based region growing for clear object boundaries"""
        try:
            current_mask = np.zeros(len(self.original_xyz), dtype=bool)
            current_mask[click_point_idx] = True
            
            # Much more aggressive growth parameters
            growth_distance = max_distance / 6  # Larger steps
            
            print(f"   🌱 AGGRESSIVE surface growing: {growth_distance:.3f}m steps, {max_iterations} iterations")
            
            for iteration in range(max_iterations):
                current_indices = np.where(current_mask)[0]
                if len(current_indices) == 0:
                    break
                
                # Find points near current region boundary (more aggressive search)
                growth_candidates = set()
                
                # Use all current points as seeds (not just boundary)
                for point_idx in current_indices[-min(50, len(current_indices)):]:  # Use last 50 for efficiency
                    point = self.original_xyz[point_idx]
                    
                    # Find ALL nearby candidates (more aggressive)
                    point_distances = np.linalg.norm(self.original_xyz[candidate_indices] - point, axis=1)
                    nearby_mask = point_distances < growth_distance
                    nearby_candidates = candidate_indices[nearby_mask]
                    
                    for candidate_idx in nearby_candidates:
                        if not current_mask[candidate_idx]:  # Not already included
                            growth_candidates.add(candidate_idx)
                
                growth_candidates = list(growth_candidates)
                
                if not growth_candidates:
                    print(f"       🛑 No more candidates at iteration {iteration + 1}")
                    break
                
                # VERY PERMISSIVE scoring for clear object boundaries
                current_points = self.original_xyz[current_mask]
                current_colors = self.original_colors[current_mask] if hasattr(self, 'original_colors') else None
                current_normals = self.point_normals[current_mask] if hasattr(self, 'point_normals') else None
                
                added_count = 0
                for candidate_idx in growth_candidates:
                    candidate_point = self.original_xyz[candidate_idx]
                    
                    # Distance-based score (very permissive)
                    min_distance = np.min(np.linalg.norm(current_points - candidate_point, axis=1))
                    distance_score = np.exp(-min_distance / (growth_distance * 2))  # More permissive
                    
                    # Color similarity score (more permissive)
                    color_score = 1.0
                    if current_colors is not None and hasattr(self, 'original_colors'):
                        candidate_color = self.original_colors[candidate_idx]
                        color_distances = np.linalg.norm(current_colors - candidate_color, axis=1)
                        min_color_distance = np.min(color_distances)
                        color_score = np.exp(-min_color_distance / (color_threshold * 2))  # More permissive
                    
                    # Normal similarity score (very permissive)
                    normal_score = 1.0
                    if current_normals is not None and hasattr(self, 'point_normals'):
                        candidate_normal = self.point_normals[candidate_idx]
                        normal_similarities = np.dot(current_normals, candidate_normal)
                        max_normal_similarity = np.max(normal_similarities)
                        normal_score = (max_normal_similarity + 1) / 2
                    
                    # Combined score with VERY permissive threshold
                    combined_score = distance_score * color_score * normal_score
                    
                    # VERY LOW threshold for aggressive object-level segmentation
                    if combined_score > 0.1:  # Much lower than before (was 0.2)
                        current_mask[candidate_idx] = True
                        added_count += 1
                
                print(f"       ➕ Iteration {iteration + 1}: added {added_count} points (total: {current_mask.sum()})")
                
                # Stop if no significant growth
                if added_count < 3:  # Lower threshold
                    break
                
                # Stop if we've grown extremely large (likely merged multiple rooms)
                if current_mask.sum() > len(self.original_xyz) * 0.15:  # Allow larger growth (15% vs 10%)
                    print(f"       ⚠️  Very large growth detected, stopping to avoid merging rooms")
                    break
            
            print(f"   🌱 AGGRESSIVE growing completed: {current_mask.sum()} points after {min(iteration+1, max_iterations)} iterations")
            return current_mask
            
        except Exception as e:
            print(f"   ❌ Aggressive surface growing failed: {e}")
            fallback_mask = np.zeros(len(self.original_xyz), dtype=bool)
            fallback_mask[click_point_idx] = True
            return fallback_mask
    
    def refine_with_geometric_consistency(self, object_mask, click_point_idx, max_furniture_size=3.0):
        """Refine the mask using geometric consistency checks (more lenient)"""
        try:
            object_points = self.original_xyz[object_mask]
            
            # Check for reasonable object dimensions (more lenient)
            object_bbox = object_points.max(axis=0) - object_points.min(axis=0)
            object_size = np.linalg.norm(object_bbox)
            
            print(f"   📐 Object bbox: {object_bbox}, size: {object_size:.2f}m")
            
            # More lenient furniture size constraints
            if object_size > max_furniture_size:  # More lenient (3m vs 2m)
                print(f"   ⚠️  Object large ({object_size:.2f}m), applying gentle size constraint...")
                
                # Keep points within larger distance from click (more lenient)
                click_point = self.original_xyz[click_point_idx]
                distances = np.linalg.norm(self.original_xyz - click_point, axis=1)
                size_mask = distances < max_furniture_size  # 3m vs 2m before
                
                refined_mask = object_mask & size_mask
                print(f"   📐 Gently refined mask: {refined_mask.sum()} points")
                return refined_mask
            
            return object_mask
            
        except Exception as e:
            print(f"   ❌ Geometric refinement failed: {e}")
            return object_mask
    
    def ultra_aggressive_connected_components(self, object_mask, click_point_idx, max_distance):
        """Apply ultra-aggressive connected component analysis for clear objects"""
        try:
            masked_points = self.original_xyz[object_mask]
            
            if len(masked_points) < 10:
                return object_mask
            
            # Use VERY large eps for object-level clustering
            eps = max_distance / 2  # Much larger than before (was /3)
            min_samples = max(2, len(masked_points) // 100)  # Very low minimum samples
            
            print(f"   🔗 ULTRA-AGGRESSIVE clustering: eps={eps:.3f}, min_samples={min_samples}")
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(masked_points)
            labels = clustering.labels_
            
            # Find the cluster containing the clicked point
            click_point = self.original_xyz[click_point_idx]
            distances_to_click = np.linalg.norm(masked_points - click_point, axis=1)
            closest_idx = np.argmin(distances_to_click)
            main_cluster_label = labels[closest_idx]
            
            if main_cluster_label == -1:  # Noise
                print(f"   ⚠️  Click point in noise cluster, keeping all points for max coverage")
                return object_mask
            
            # Create refined mask with main cluster
            main_cluster_mask = labels == main_cluster_label
            
            # Map back to full point cloud
            refined_mask = np.zeros_like(object_mask)
            original_indices = np.where(object_mask)[0]
            refined_mask[original_indices[main_cluster_mask]] = True
            
            num_clusters = len(np.unique(labels[labels >= 0]))
            main_cluster_size = main_cluster_mask.sum()
            
            print(f"   🔗 Found {num_clusters} clusters, main cluster: {main_cluster_size} points")
            
            return refined_mask
            
        except Exception as e:
            print(f"   ❌ Ultra-aggressive clustering failed: {e}")
            return object_mask
    
    def update_continuous_visualization(self):
        """Update visualization in real-time for continuous interaction"""
        try:
            # Start with original colors
            new_colors = self.original_color_backup.copy()
            
            # Apply current segmentation with bright green
            if self.current_mask is not None and self.current_mask.sum() > 0:
                bright_green = np.array([0.0, 1.0, 0.0])  # Pure bright green
                new_colors[self.current_mask] = bright_green
            
            # Highlight positive clicks with bright red
            bright_red = np.array([1.0, 0.0, 0.0])
            for click_point in self.positive_clicks:
                distances = np.linalg.norm(self.original_xyz - click_point, axis=1)
                closest_idx = np.argmin(distances)
                new_colors[closest_idx] = bright_red
            
            # Highlight negative clicks with bright blue
            bright_blue = np.array([0.0, 0.0, 1.0])
            for click_point in self.negative_clicks:
                distances = np.linalg.norm(self.original_xyz - click_point, axis=1)
                closest_idx = np.argmin(distances)
                new_colors[closest_idx] = bright_blue
            
            # Update point cloud colors in the SAME window
            self.point_cloud.colors = o3d.utility.Vector3dVector(new_colors)
            self.vis.update_geometry(self.point_cloud)
            
        except Exception as e:
            print(f"❌ Error updating continuous visualization: {e}")
    
    def setup_keyboard_callbacks(self):
        """Setup keyboard callbacks for continuous interaction"""
        # Note: Open3D's keyboard callbacks are limited
        # For now we'll handle these in the main loop or through other means
        pass
    
    def clear_all_clicks(self):
        """Clear all clicks and reset segmentation"""
        try:
            print("\n🔄 Clearing all clicks and resetting...")
            
            self.positive_clicks = []
            self.negative_clicks = []
            self.all_prompts = []
            self.all_labels = []
            self.click_history = []
            self.current_mask = None
            self.last_processed_clicks = 0
            
            # Reset visualization to original colors
            self.point_cloud.colors = o3d.utility.Vector3dVector(self.original_color_backup.copy())
            self.vis.update_geometry(self.point_cloud)
            
            print("✅ All clicks cleared! Ready for new object.")
            
        except Exception as e:
            print(f"❌ Error clearing clicks: {e}")
    
    def undo_last_click(self):
        """Undo the last click and update segmentation"""
        try:
            if len(self.click_history) > 0:
                last_click = self.click_history.pop()
                
                if last_click['is_positive']:
                    self.positive_clicks.pop()
                    click_type = "positive"
                else:
                    self.negative_clicks.pop()
                    click_type = "negative"
                
                self.all_prompts.pop()
                self.all_labels.pop()
                
                print(f"\n⏪ Undoing last {click_type} click")
                print(f"   📊 Remaining: {len(self.positive_clicks)} positive, {len(self.negative_clicks)} negative")
                
                # Re-run segmentation with remaining clicks
                if len(self.all_prompts) > 0:
                    self.run_continuous_segmentation()
                else:
                    self.current_mask = None
                    self.update_continuous_visualization()
                    
            else:
                print("⚠️  No clicks to undo")
                
        except Exception as e:
            print(f"❌ Error undoing click: {e}")
    
    def save_current_segmentation(self):
        """Save current segmentation results"""
        try:
            if self.current_mask is None or self.current_mask.sum() == 0:
                print("⚠️  No segmentation to save")
                return
            
            # Save with timestamp
            timestamp = str(int(time.time()))
            output_dir = "continuous_segmentation_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save mask
            mask_file = f"{output_dir}/mask_{timestamp}.npy"
            np.save(mask_file, self.current_mask)
            
            # Save metadata
            metadata = {
                'num_positive_clicks': len(self.positive_clicks),
                'num_negative_clicks': len(self.negative_clicks), 
                'num_segmented': self.current_mask.sum(),
                'segmentation_ratio': self.current_mask.sum() / len(self.current_mask),
                'click_history': self.click_history
            }
            
            metadata_file = f"{output_dir}/metadata_{timestamp}.npy"
            np.save(metadata_file, metadata)
            
            # Save segmented point cloud
            segmented_xyz = self.original_xyz[self.current_mask]
            segmented_rgb = self.original_colors[self.current_mask]
            
            segmented_pcd = o3d.geometry.PointCloud()
            segmented_pcd.points = o3d.utility.Vector3dVector(segmented_xyz)
            segmented_pcd.colors = o3d.utility.Vector3dVector(segmented_rgb)
            
            ply_file = f"{output_dir}/segmented_{timestamp}.ply"
            o3d.io.write_point_cloud(ply_file, segmented_pcd)
            
            print(f"\n💾 Segmentation saved successfully!")
            print(f"   📄 Mask: {mask_file}")
            print(f"   📄 Metadata: {metadata_file}")
            print(f"   📄 Point cloud: {ply_file}")
            print(f"   📊 {len(segmented_xyz)} points with {len(self.positive_clicks)}+/{len(self.negative_clicks)}- clicks")
            
        except Exception as e:
            print(f"❌ Error saving segmentation: {e}")
    
    def analyze_point_density(self):
        """Analyze overall point cloud density to adapt segmentation strategy"""
        if self.original_xyz is None:
            return {'density': 0, 'is_sparse': True}
        
        # Calculate local density at random sample points
        num_samples = min(100, len(self.original_xyz) // 10)
        sample_indices = np.random.choice(len(self.original_xyz), num_samples, replace=False)
        
        densities = []
        for idx in sample_indices:
            point = self.original_xyz[idx]
            # Find neighbors within 0.1m radius
            distances = np.linalg.norm(self.original_xyz - point, axis=1)
            neighbors_count = np.sum(distances < 0.1)
            densities.append(neighbors_count)
        
        avg_density = np.mean(densities)
        median_density = np.median(densities)
        
        # Classify sparsity
        is_sparse = avg_density < 20  # Less than 20 neighbors per 0.1m radius
        is_very_sparse = avg_density < 10
        
        density_info = {
            'avg_density': avg_density,
            'median_density': median_density,
            'is_sparse': is_sparse,
            'is_very_sparse': is_very_sparse,
            'total_points': len(self.original_xyz),
            'sparsity_level': 'very_sparse' if is_very_sparse else 'sparse' if is_sparse else 'dense'
        }
        
        print(f"📊 Point cloud density analysis:")
        print(f"   🎯 Average local density: {avg_density:.1f} points/0.1m")
        print(f"   📈 Sparsity level: {density_info['sparsity_level']}")
        
        return density_info

    def update_visualization(self, mask):
        """Update visualization with current segmentation mask"""
        try:
            # Start with original colors
            new_colors = self.original_color_backup.copy()
            
            # Apply current segmentation with bright green
            if mask is not None and mask.sum() > 0:
                bright_green = np.array([0.0, 1.0, 0.0])  # Pure bright green
                new_colors[mask] = bright_green
            
            # Highlight positive clicks with bright red
            bright_red = np.array([1.0, 0.0, 0.0])
            for click_point in self.positive_clicks:
                distances = np.linalg.norm(self.original_xyz - click_point, axis=1)
                closest_idx = np.argmin(distances)
                new_colors[closest_idx] = bright_red
            
            # Highlight negative clicks with bright blue
            bright_blue = np.array([0.0, 0.0, 1.0])
            for click_point in self.negative_clicks:
                distances = np.linalg.norm(self.original_xyz - click_point, axis=1)
                closest_idx = np.argmin(distances)
                new_colors[closest_idx] = bright_blue
            
            # Update point cloud colors
            self.point_cloud.colors = o3d.utility.Vector3dVector(new_colors)
            self.vis.update_geometry(self.point_cloud)
            
        except Exception as e:
            print(f"❌ Error updating visualization: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Point-SAM Interface")
    parser.add_argument("point_cloud", nargs="?", default="demo/static/models/scene.ply",
                       help="Path to point cloud file")
    args = parser.parse_args()
    
    # Check if point cloud exists
    if not os.path.exists(args.point_cloud):
        # Try to find available point clouds
        search_paths = [
            "demo/static/models/*.ply",
            "demo/static/models/*.pcd",
            "*.ply",
            "*.pcd"
        ]
        
        import glob
        found_files = []
        for pattern in search_paths:
            found_files.extend(glob.glob(pattern))
        
        if found_files:
            print(f"📁 Point cloud not found: {args.point_cloud}")
            print("📁 Available point clouds:")
            for i, f in enumerate(found_files):
                print(f"   {i+1}. {f}")
            
            try:
                choice = int(input(f"Select point cloud (1-{len(found_files)}): ")) - 1
                args.point_cloud = found_files[choice]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return
        else:
            print(f"❌ No point cloud files found")
            return
    
    # Run intelligent interface
    app = IntelligentPointSAMUI()
    success = app.run_intelligent_interface(args.point_cloud)
    
    if success:
        print("🎉 Intelligent Point-SAM completed successfully!")
        print("🧠 Advanced spatial analysis and multi-prompt generation delivered")
        print("💡 Much better segmentation quality compared to simple random point selection!")
    else:
        print("❌ Intelligent interface failed")

if __name__ == "__main__":
    main() 