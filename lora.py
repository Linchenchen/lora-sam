import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def apply_lora(parent_block: nn.Module, device):
    for name, block in parent_block.named_children():
        if isinstance(block, nn.Linear):
            block = MonkeyPatchLoRALinear(block, 4, 1).to(device)
            setattr(parent_block, name, block)

        elif isinstance(block, nn.Conv2d):
            block = MonkeyPatchLoRAConv2D(block, 4, 1).to(device)
            setattr(parent_block, name, block)

        elif isinstance(block, nn.ConvTranspose2d):
            block = MonkeyPatchLoRAConvTranspose2D(block, 4, 1).to(device)
            setattr(parent_block, name, block)

        elif isinstance(block, nn.Module):
            apply_lora(block, device=device)


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)

    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return 0

class MonkeyPatchLoRALinear(nn.Module):
    # It's "monkey patch" means you can replace nn.Linear with the new
    # LoRA Linear class without modifying any other code.
    def __init__(self, fc: nn.Linear, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(fc.in_features, fc.out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(fc.in_features, fc.out_features)}"
            )
        if not isinstance(fc, nn.Linear):
            raise ValueError(
                f"MonkeyPatchLoRALinear only support nn.Linear, but got {type(fc)}"
            )

        self.fc = fc
        self.rank = rank
        self.lora_scale = lora_scale

        in_features = fc.in_features
        out_features = fc.out_features
        self.fc_lora = LoRALinearLayer(in_features, out_features, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc(hidden_states) + \
                        self.lora_scale * self.fc_lora(hidden_states)
        return hidden_states

    @property
    def weight(self):
        return self.fc.weight + self.lora_scale * self.fc_lora.weight

    @property
    def bias(self):
        return self.fc.bias

# your implementation

class LoRAConv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, rank=4):
        super().__init__()
        # if rank > min(in_channels * kernel_size**2, out_channels):
        #     raise ValueError(
        #         f"LoRA rank {rank} must be less or equal than {min(in_channels * kernel_size**2, out_channels)}"
        #     )

        self.down = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.up = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = F.conv2d(hidden_states.to(dtype), self.down.weight, self.down.bias, self.down.stride, self.down.padding)
        up_hidden_states = F.conv2d(down_hidden_states, self.up.weight, self.up.bias, self.up.stride, self.up.padding)

        return up_hidden_states.to(orig_dtype)
    
    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return 0

class MonkeyPatchLoRAConv2D(nn.Module):
    def __init__(self, conv2d: nn.Conv2d, rank=4, lora_scale=1):
        super().__init__()
        # if rank > min(conv2d.in_channels * conv2d.kernel_size[0]**2, conv2d.out_channels):
        #     raise ValueError(
        #         f"LoRA rank {rank} must be less or equal than {min(conv2d.in_channels * conv2d.kernel_size[0]**2, conv2d.out_channels)}"
        #     )
        if not isinstance(conv2d, nn.Conv2d):
            raise ValueError(
                f"MonkeyPatchLoRAConv2D only supports nn.Conv2d, but got {type(conv2d)}"
            )

        self.conv2d = conv2d
        self.rank = rank
        self.lora_scale = lora_scale

        in_channels = conv2d.in_channels
        out_channels = conv2d.out_channels
        kernel_size = conv2d.kernel_size
        stride = conv2d.stride
        padding = conv2d.padding
        self.conv2d_lora = LoRAConv2DLayer(in_channels, out_channels, kernel_size, stride, padding, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv2d(hidden_states) + \
                        self.lora_scale * self.conv2d_lora(hidden_states)
        return hidden_states

    @property
    def weight(self):
        return self.conv2d.weight + self.lora_scale * self.conv2d_lora.weight

    @property
    def bias(self):
        return self.conv2d.bias

class LoRAConvTranspose2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, rank=4):
        super().__init__()

        if rank > min(in_channels, out_channels * kernel_size**2):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_channels, out_channels * kernel_size**2)}"
            )

        # Define the downsample convolution layer
        self.down = nn.ConvTranspose2d(in_channels, rank, kernel_size, stride, padding, output_padding, bias=False)
        
        # Define the upsample convolution layer
        self.up = nn.ConvTranspose2d(rank, out_channels, 1, 1, 0, bias=False)

        # Initialize weights
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        # Perform the downsample convolution
        down_hidden_states = F.conv_transpose2d(hidden_states.to(dtype), self.down.weight, self.down.bias, self.down.stride, self.down.padding, self.down.output_padding)
        
        # Perform the upsample convolution
        up_hidden_states = F.conv_transpose2d(down_hidden_states, self.up.weight, self.up.bias, self.up.stride, self.up.padding, self.up.output_padding)

        return up_hidden_states.to(orig_dtype)

    @property
    def weight(self):
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return 0


class MonkeyPatchLoRAConvTranspose2D(nn.Module):
    def __init__(self, conv_transpose2d: nn.ConvTranspose2d, rank=4, lora_scale=1):
        super().__init__()
        if rank > min(conv_transpose2d.in_channels, conv_transpose2d.out_channels * conv_transpose2d.kernel_size[0]**2):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(conv_transpose2d.in_channels, conv_transpose2d.out_channels * conv_transpose2d.kernel_size[0]**2)}"
            )
        if not isinstance(conv_transpose2d, nn.ConvTranspose2d):
            raise ValueError(
                f"MonkeyPatchLoRAConvTranspose2D only supports nn.ConvTranspose2d, but got {type(conv_transpose2d)}"
            )

        # Save the original ConvTranspose2D layer
        self.conv_transpose2d = conv_transpose2d
        self.rank = rank
        self.lora_scale = lora_scale

        # Create the LoRA ConvTranspose2D layer
        in_channels = conv_transpose2d.in_channels
        out_channels = conv_transpose2d.out_channels
        kernel_size = conv_transpose2d.kernel_size
        stride = conv_transpose2d.stride
        padding = conv_transpose2d.padding
        output_padding = conv_transpose2d.output_padding
        self.conv_transpose2d_lora = LoRAConvTranspose2DLayer(in_channels, out_channels, kernel_size, stride, padding, output_padding, rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Forward pass using both the original ConvTranspose2D and LoRA ConvTranspose2D
        hidden_states = self.conv_transpose2d(hidden_states) + \
                        self.lora_scale * self.conv_transpose2d_lora(hidden_states)
        return hidden_states

    @property
    def weight(self):
        # Combine weights of the original ConvTranspose2D and LoRA ConvTranspose2D
        return self.conv_transpose2d.weight + self.lora_scale * self.conv_transpose2d_lora.weight

    @property
    def bias(self):
        # Return the bias of the original ConvTranspose2D
        return self.conv_transpose2d.bias