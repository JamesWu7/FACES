import matplotlib.pyplot as plt
import os
import numpy as np


def visualize_activations(activations, save_dir, sample_name=None):
    """可视化激活图并保存到指定样本目录"""
    try:
        os.makedirs(save_dir, exist_ok=True)

        conv_activations = {}
        attention_activations = {}
        cbam_activations = {}

        for name, activation in activations.items():
            if 'conv' in name.lower() and 'attention' not in name.lower():
                conv_activations[name] = activation
            elif 'attention' in name.lower():
                attention_activations[name] = activation
            elif 'cbam' in name.lower():
                cbam_activations[name] = activation

        if conv_activations:
            _visualize_conv_features_3d(conv_activations, save_dir, sample_name)

        if attention_activations:
            _visualize_attention_weights_3d(attention_activations, save_dir, sample_name)

        if cbam_activations:
            _visualize_cbam_output_3d(cbam_activations, save_dir, sample_name)

        print(f"激活图已保存至 {save_dir}")

    except Exception as e:
        print(f"可视化激活图时出错: {e}")
        import traceback
        traceback.print_exc()


def _build_title(prefix, layer_name, sample_name=None):
    if sample_name:
        return f'{prefix} - {layer_name} ({sample_name})'
    return f'{prefix} - {layer_name}'


def _visualize_conv_features_3d(activations, save_dir, sample_name=None, num_channels=8):
    """可视化3D卷积层特征图"""
    for layer_name, activation in activations.items():
        if activation.ndim == 3:
            num_total_channels, h, w = activation.shape
            activation_np = activation.detach().cpu().numpy()
        elif activation.ndim == 4:
            _, num_total_channels, h, w = activation.shape
            activation_np = activation[0].detach().cpu().numpy()
        else:
            print(f"跳过 {layer_name}，不支持的维度: {activation.ndim}")
            continue

        num_display = min(num_channels, num_total_channels)
        channels_to_show = np.linspace(0, num_total_channels - 1, num_display, dtype=int)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for idx, channel in enumerate(channels_to_show):
            if idx < len(axes):
                ax = axes[idx]
                channel_data = activation_np[channel]

                if channel_data.max() > channel_data.min():
                    channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
                else:
                    channel_data = np.zeros_like(channel_data)

                im = ax.imshow(channel_data, cmap='viridis')
                ax.set_title(f'{layer_name}\nCh{channel} ({h}x{w})', fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, shrink=0.6)

        for idx in range(len(channels_to_show), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(_build_title('Conv Features', layer_name, sample_name), fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'conv_{layer_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def _visualize_attention_weights_3d(activations, save_dir, sample_name=None):
    """可视化3D注意力权重"""
    for layer_name, activation in activations.items():
        if activation.ndim == 3:
            c, h, w = activation.shape
            activation_np = activation.detach().cpu().numpy()
        elif activation.ndim == 4:
            _, c, h, w = activation.shape
            activation_np = activation[0].detach().cpu().numpy()
        else:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if 'channel' in layer_name.lower():
            if c > 1 and h == 1 and w == 1:
                channel_weights = activation_np.squeeze()
                axes[0].bar(range(len(channel_weights)), channel_weights, color='skyblue')
                axes[0].set_title(f'{layer_name}\nChannel Weights')
                axes[0].set_xlabel('Channel Index')
                axes[0].set_ylabel('Weight')
                axes[0].grid(True, alpha=0.3)

                axes[1].hist(channel_weights, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[1].set_title('Weight Distribution')
                axes[1].set_xlabel('Weight Value')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(True, alpha=0.3)

                axes[2].text(0.1, 0.8, f'Min: {channel_weights.min():.4f}\n'
                                       f'Max: {channel_weights.max():.4f}\n'
                                       f'Mean: {channel_weights.mean():.4f}\n'
                                       f'Std: {channel_weights.std():.4f}',
                             transform=axes[2].transAxes, fontsize=12, verticalalignment='top')
                axes[2].set_title('Statistics')
                axes[2].axis('off')

        elif 'spatial' in layer_name.lower():
            if c == 1:
                spatial_weights = activation_np[0]

                im1 = axes[0].imshow(spatial_weights, cmap='hot')
                axes[0].set_title(f'{layer_name}\nOriginal\n{spatial_weights.shape}')
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0], shrink=0.8)

                if spatial_weights.max() > spatial_weights.min():
                    enhanced = (spatial_weights - spatial_weights.min()) / (
                        spatial_weights.max() - spatial_weights.min())
                    enhanced = np.power(enhanced, 0.3)
                else:
                    enhanced = spatial_weights

                im2 = axes[1].imshow(enhanced, cmap='hot')
                axes[1].set_title('Enhanced Contrast')
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1], shrink=0.8)

                axes[2].text(0.1, 0.8, f'Min: {spatial_weights.min():.4f}\n'
                                       f'Max: {spatial_weights.max():.4f}\n'
                                       f'Mean: {spatial_weights.mean():.4f}\n'
                                       f'Shape: {spatial_weights.shape}',
                             transform=axes[2].transAxes, fontsize=10, verticalalignment='top')
                axes[2].set_title('Statistics')
                axes[2].axis('off')

        plt.suptitle(_build_title('Attention Weights', layer_name, sample_name), fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'attention_{layer_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def _visualize_cbam_output_3d(activations, save_dir, sample_name=None):
    """可视化CBAM输出"""
    for layer_name, activation in activations.items():
        if activation.ndim == 3:
            c, h, w = activation.shape
            activation_np = activation.detach().cpu().numpy()
        elif activation.ndim == 4:
            _, c, h, w = activation.shape
            activation_np = activation[0].detach().cpu().numpy()
        else:
            continue

        num_display = min(8, c)
        channels_to_show = np.linspace(0, c - 1, num_display, dtype=int)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for idx, channel in enumerate(channels_to_show):
            if idx < len(axes):
                ax = axes[idx]
                channel_data = activation_np[channel]

                if channel_data.max() > channel_data.min():
                    channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())

                im = ax.imshow(channel_data, cmap='viridis')
                ax.set_title(f'{layer_name}\nCh{channel}', fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, shrink=0.6)

        for idx in range(len(channels_to_show), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(_build_title('CBAM Output', layer_name, sample_name), fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'cbam_{layer_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
