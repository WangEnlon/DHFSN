#!/usr/bin/env python3
"""
Example script demonstrating how to use DHFSN for image dehazing.
"""

import os
import torch
import argparse
from model import build_net
from torchvision.transforms import functional as F
from PIL import Image
import torch.nn.functional as f


def dehaze_single_image(model, image_path, output_path=None):
    """
    Dehaze a single image using the trained DHFSN model.

    Args:
        model: Trained DHFSN model
        image_path: Path to the hazy input image
        output_path: Path to save the dehazed image (optional)

    Returns:
        Dehazed image as PIL Image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Pad image to be divisible by 8
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    factor = 8
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = f.pad(input_tensor, (0, padw, 0, padh), 'reflect')

    # Run inference
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)[2]  # Use the finest scale output
        output = output[:, :, :h, :w]  # Remove padding

    # Post-process
    output = torch.clamp(output, 0, 1)
    output_image = F.to_pil_image(output.squeeze(0).cpu())

    # Save if output path is provided
    if output_path:
        output_image.save(output_path)
        print(f"Dehazed image saved to: {output_path}")

    return output_image


def main():
    parser = argparse.ArgumentParser(description='DHFSN Single Image Dehazing Example')
    parser.add_argument('--input', type=str, required=True, help='Path to hazy input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save dehazed image')
    parser.add_argument('--model', type=str, default='results/Best.pkl', help='Path to trained model')

    args = parser.parse_args()

    # Check if input image exists
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        return

    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_dehazed.png"

    # Build and load model
    print("Loading DHFSN model...")
    model = build_net()

    if os.path.exists(args.model):
        state_dict = torch.load(args.model)
        model.load_state_dict(state_dict['model'])
        print(f"Model loaded from: {args.model}")
    else:
        print(f"Warning: Model file not found: {args.model}")
        print("Using untrained model for demonstration purposes.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Dehaze image
    print(f"Dehazing image: {args.input}")
    dehaze_single_image(model, args.input, args.output)
    print("Done!")


if __name__ == '__main__':
    main()
