"""
Visualization script for token activation analysis
Usage: python visualize_activations.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def load_activation_data(filepath):
    """Load saved activation data."""
    data = np.load(filepath, allow_pickle=True).item()
    return data

def visualize_activations(data, save_path='/root/LLaVA-MoICE/llava/model/language_model/llama/activation/visualization.png'):
    """
    Create comprehensive visualization of token activations.
    
    Args:
        data: Dictionary containing:
            - activations: 1D array of activation values
            - image_start: Start index of image tokens
            - image_len: Length of image region
            - seq_len: Total sequence length
            - layer_idx: Layer index
    """
    activations = data['activations']
    image_start = data['image_start']
    image_len = data['image_len']
    seq_len = data['seq_len']
    layer_idx = data['layer_idx']
    
    image_end = image_start + image_len
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ==================== Plot 1: Full Sequence Line Plot ====================
    ax1 = fig.add_subplot(gs[0, :])
    
    token_indices = np.arange(seq_len)
    ax1.plot(token_indices, activations, 'b-', linewidth=0.8, alpha=0.7, label='All Tokens')
    
    # Highlight image region
    ax1.axvspan(image_start, image_end, alpha=0.3, color='orange', label='Image Region')
    
    # Mark boundaries
    ax1.axvline(x=image_start, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Image Start')
    ax1.axvline(x=image_end, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Image End')
    
    ax1.set_xlabel('Token Position', fontsize=12)
    ax1.set_ylabel('Mean Activation (RMS norm)', fontsize=12)
    ax1.set_title(f'Layer {layer_idx}: Token Activations Across Sequence', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ==================== Plot 2: Zoomed Image Region ====================
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Extract regions
    text_before = activations[:image_start] if image_start > 0 else np.array([])
    image_tokens = activations[image_start:image_end]
    text_after = activations[image_end:] if image_end < seq_len else np.array([])
    
    # Plot zoomed view around image region
    zoom_margin = min(50, image_start, seq_len - image_end)
    zoom_start = max(0, image_start - zoom_margin)
    zoom_end = min(seq_len, image_end + zoom_margin)
    
    zoom_indices = np.arange(zoom_start, zoom_end)
    zoom_activations = activations[zoom_start:zoom_end]
    
    ax2.plot(zoom_indices, zoom_activations, 'b-', linewidth=1.2, alpha=0.7)
    ax2.axvspan(image_start, image_end, alpha=0.3, color='orange')
    ax2.axvline(x=image_start, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=image_end, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Token Position', fontsize=11)
    ax2.set_ylabel('Mean Activation', fontsize=11)
    ax2.set_title('Zoomed View: Image Region', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ==================== Plot 3: Distribution Histogram ====================
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create histogram for each region
    bins = 50
    
    if len(text_before) > 0:
        ax3.hist(text_before, bins=bins, alpha=0.6, label=f'Text Before ({len(text_before)})', 
                 color='blue', density=True)
    
    ax3.hist(image_tokens, bins=bins, alpha=0.6, label=f'Image ({len(image_tokens)})', 
             color='orange', density=True)
    
    if len(text_after) > 0:
        ax3.hist(text_after, bins=bins, alpha=0.6, label=f'Text After ({len(text_after)})', 
                 color='green', density=True)
    
    ax3.set_xlabel('Activation Value', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Activation Distribution by Region', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ==================== Plot 4: Box Plot Comparison ====================
    ax4 = fig.add_subplot(gs[2, 0])
    
    data_to_plot = []
    labels = []
    
    if len(text_before) > 0:
        data_to_plot.append(text_before)
        labels.append(f'Text Before\n({len(text_before)})')
    
    data_to_plot.append(image_tokens)
    labels.append(f'Image\n({len(image_tokens)})')
    
    if len(text_after) > 0:
        data_to_plot.append(text_after)
        labels.append(f'Text After\n({len(text_after)})')
    
    bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'orange', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Activation Value', fontsize=11)
    ax4.set_title('Statistical Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ==================== Plot 5: Statistics Table ====================
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate statistics
    stats_data = []
    
    if len(text_before) > 0:
        text_all = np.concatenate([text_before, text_after]) if len(text_after) > 0 else text_before
    elif len(text_after) > 0:
        text_all = text_after
    else:
        text_all = np.array([])
    
    # Build table data
    if len(text_all) > 0:
        stats_data.append(['Metric', 'Text Tokens', 'Image Tokens', 'Ratio (Img/Text)'])
        stats_data.append(['Count', f'{len(text_all)}', f'{len(image_tokens)}', '-'])
        stats_data.append(['Mean', f'{text_all.mean():.6f}', f'{image_tokens.mean():.6f}', 
                          f'{image_tokens.mean()/text_all.mean():.4f}x'])
        stats_data.append(['Std', f'{text_all.std():.6f}', f'{image_tokens.std():.6f}', 
                          f'{image_tokens.std()/text_all.std():.4f}x'])
        stats_data.append(['Min', f'{text_all.min():.6f}', f'{image_tokens.min():.6f}', 
                          f'{image_tokens.min()/text_all.min():.4f}x'])
        stats_data.append(['Max', f'{text_all.max():.6f}', f'{image_tokens.max():.6f}', 
                          f'{image_tokens.max()/text_all.max():.4f}x'])
    else:
        stats_data.append(['Metric', 'Image Tokens'])
        stats_data.append(['Count', f'{len(image_tokens)}'])
        stats_data.append(['Mean', f'{image_tokens.mean():.6f}'])
        stats_data.append(['Std', f'{image_tokens.std():.6f}'])
        stats_data.append(['Min', f'{image_tokens.min():.6f}'])
        stats_data.append(['Max', f'{image_tokens.max():.6f}'])
    
    table = ax5.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25] if len(text_all) > 0 else [0.4, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Detailed Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # ==================== Overall Title ====================
    fig.suptitle(f'Token Activation Analysis - Layer {layer_idx}\n'
                 f'Total Tokens: {seq_len} | Image Region: [{image_start}, {image_end})',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    
    # Also save a simplified version
    simple_path = save_path.replace('.png', '_simple.png')
    
    fig_simple, ax = plt.subplots(figsize=(14, 4))
    ax.plot(token_indices, activations, 'b-', linewidth=1, alpha=0.7)
    ax.axvspan(image_start, image_end, alpha=0.3, color='orange', label='Image Region')
    ax.axvline(x=image_start, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=image_end, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Mean Activation (RMS norm)', fontsize=12)
    ax.set_title(f'Layer {layer_idx}: Token Activations - Image vs Text', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(simple_path, dpi=300, bbox_inches='tight')
    print(f"✅ Simple visualization saved to: {simple_path}")
    
    plt.close('all')
    
    return fig


def print_summary(data):
    """Print text summary of the analysis."""
    activations = data['activations']
    image_start = data['image_start']
    image_len = data['image_len']
    seq_len = data['seq_len']
    
    image_end = image_start + image_len
    
    text_before = activations[:image_start] if image_start > 0 else np.array([])
    image_tokens = activations[image_start:image_end]
    text_after = activations[image_end:] if image_end < seq_len else np.array([])
    
    print("\n" + "="*80)
    print("📊 ACTIVATION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n🔍 Sequence Structure:")
    print(f"   Total tokens: {seq_len}")
    print(f"   Text before image: {len(text_before)} tokens")
    print(f"   Image tokens: {len(image_tokens)} tokens ({image_start} to {image_end})")
    print(f"   Text after image: {len(text_after)} tokens")
    
    if len(text_before) > 0 and len(text_after) > 0:
        text_all = np.concatenate([text_before, text_after])
    elif len(text_before) > 0:
        text_all = text_before
    elif len(text_after) > 0:
        text_all = text_after
    else:
        text_all = np.array([])
    
    if len(text_all) > 0:
        print(f"\n📈 Key Findings:")
        print(f"   Image tokens have {image_tokens.mean()/text_all.mean():.2f}x higher mean activation than text")
        print(f"   Image max is {image_tokens.max()/text_all.max():.2f}x higher than text max")
        
        if image_tokens.mean() > text_all.mean():
            print(f"   ✅ Image tokens show HIGHER activation (as expected for sink tokens)")
        else:
            print(f"   ⚠️ Image tokens show LOWER activation (unexpected)")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Check if data file exists
    data_file = '/root/LLaVA-MoICE/llava/model/language_model/llama/activation/layer0.npy'
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        print("Please run your training script first to generate the data.")
    else:
        # Load data
        print("Loading activation data...")
        data = load_activation_data(data_file)
        
        # Print summary
        print_summary(data)
        
        # Create visualizations
        print("Creating visualizations...")
        visualize_activations(data)
        
        print("\n✅ All done! Check the generated PNG files.")