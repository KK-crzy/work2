import os

import cv2
import torch
import numpy as np
import argparse
from networks.srn.net import SpeckleReductionNet
from utils.image_ops import linear_normalization


def load_model(model_path, device):
    model = SpeckleReductionNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def apply_gamma(image, gamma_value):
    """
    Apply gamma correction to a normalized image (0~1 or 0~255).
    """
    image = np.clip(image, 0, 1)
    corrected = np.power(image, gamma_value)
    return (corrected * 255).astype(np.uint8)

def visualize_result(norm_input, output):
    """
    Display input and gamma-corrected output in two separate windows.
    Gamma is adjustable from 1.0 to 3.0 (step 0.1) using a short slider.
    Gamma correction helps align denoised results with human perception,
    but ideal values may vary across ultrasound device types.
    """
    win_input = "Input"
    win_output = "Output (Gamma corrected)"
    slider_name = 'Gamma x10'

    cv2.namedWindow(win_input, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_output, cv2.WINDOW_NORMAL)

    # Move windows to avoid overlap
    cv2.moveWindow(win_input, 100, 100)
    cv2.moveWindow(win_output, 750, 100)

    def nothing(x): pass

    # Create short gamma slider: 10-30 → gamma 1.0–3.0
    if cv2.getTrackbarPos(slider_name, win_output) == -1:
        cv2.createTrackbar(slider_name, win_output, 15, 30, nothing) # Default to 1.5 (15/10)

    # Show input image
    input_uint8 = (np.clip(norm_input, 0, 1) * 255).astype(np.uint8)
    cv2.imshow(win_input, input_uint8)

    while True:
        slider_val = cv2.getTrackbarPos(slider_name, win_output)
        gamma = max(slider_val / 10.0, 1.0)

        output_clipped = np.clip(output, 0, 1)
        output_gamma = np.power(output_clipped, gamma)
        output_uint8 = (output_gamma * 255).astype(np.uint8)

        cv2.imshow(win_output, output_uint8)

        key = cv2.waitKey(1)
        if key != -1:
            break

    cv2.destroyWindow(win_input)
    cv2.destroyWindow(win_output)

    return output_gamma



def run_inference(model, image_array, device, visualize=False):
    output_list = []

    with torch.no_grad():
        for i, img_input in enumerate(image_array):
            norm_input = linear_normalization(img_input)
            tensor_input = torch.tensor(norm_input).unsqueeze(0).unsqueeze(0).to(device)

            output_tensor, _, _ = model(tensor_input, tensor_input, tensor_input)
            output_tensor = torch.clamp(output_tensor, 0, 1)
            output = output_tensor.squeeze().cpu().numpy()

            if visualize:
                # visualize and output in [0,1]
                output = visualize_result(norm_input, output)

            output_list.append(output)

    return np.stack(output_list, axis=0)


def save_results(output_array, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, output_array)
    print(f"[✓] Output saved to: {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    images = np.load(args.data_path)
    # Handle both (N, H, W) and (N, 2, H, W) input formats
    if images.ndim == 4 and images.shape[1] == 2:
        noisy_images = images[:, 0]
        gt_images = images[:, 1]
        print(f"[INFO] Loaded paired dataset: noisy shape {noisy_images.shape}, GT shape {gt_images.shape}")
    else:
        noisy_images = images
        gt_images = None
        print(f"[INFO] Loaded noisy-only dataset: shape {noisy_images.shape}")

    outputs = run_inference(model, noisy_images, device, visualize=args.visualize)

    save_results(outputs, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SRN model inference.")
    parser.add_argument('--data_path', type=str, default="/data2/user24/cz/speckle2self-main/data/inVivo/test_data.npy", required=True, help="Path to .npy input image file")
    parser.add_argument('--model_path', type=str, default="/data2/user24/cz/speckle2self-main/data/inVivo/model_2833.pth", required=True, help="Path to model .pth file")
    parser.add_argument('--output_path', type=str, required=True, help="Where to save the output .npy file")
    parser.add_argument('--visualize', action='store_true', help="Whether to visualize each result")
    args = parser.parse_args()

    main(args)