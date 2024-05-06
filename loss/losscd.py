import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, generate_binary_structure

'''
We provide a straightforward implementation of a "Loss-Curve-and-Data" method. 
This implementation calculates and accumulates the "Loss-Curve" and "Loss-Data" by applying them separately to a binarized image.

Computation of energy rsf follows the original paper:
Minimization of Region-Scalable Fitting Energy for Image Segmentation

We provide the implementation of the region-scalable fitting energy function in the paper.
Denoted as Loss-Data
'''
def compute_f1_f2(image, phi, sigma):
    H_phi = np.heaviside(phi, 1.0)
    weighted_image_inside = gaussian_filter(image * H_phi, sigma=sigma)
    weighted_image_outside = gaussian_filter(image * (1 - H_phi), sigma=sigma)
    normalization_inside = gaussian_filter(H_phi, sigma=sigma)
    normalization_outside = gaussian_filter(1 - H_phi, sigma=sigma)
    epsilon = 1e-10
    f1 = weighted_image_inside / (normalization_inside + epsilon)
    f2 = weighted_image_outside / (normalization_outside + epsilon)
    return f1, f2

def compute_total_energy(image, segmentation, f1, f2, lambda1=1, lambda2=1, nu=0.1):
    term_inside = lambda1 * (image - f1)**2 * segmentation
    term_outside = lambda2 * (image - f2)**2 * (1 - segmentation)
    rsf_energy = np.sum(term_inside + term_outside)
    struct = generate_binary_structure(2, 1)
    boundary = binary_dilation(segmentation, structure=struct) & ~binary_erosion(segmentation, structure=struct)
    length_energy = nu * np.sum(boundary)
    total_energy = rsf_energy + length_energy
    return total_energy

'''
We provide a simple implementation of the curvature term denoted as Loss-Curve
A better computation of high accuracy curvature can be found in the following paper:
Minimizing Discrete Total Curvature for Image Processing,
which is recommended in our DDPNeXt paper.
'''
def compute_curvature_sum(segmentation):
    struct = generate_binary_structure(2, 1)
    dilated = binary_dilation(segmentation, structure=struct)
    eroded = binary_erosion(segmentation, structure=struct)
    boundary = dilated & ~eroded
    total_curvature = 0
    for i in range(1, boundary.shape[0] - 1):
        for j in range(1, boundary.shape[1] - 1):
            if boundary[i, j]:
                count = np.sum(boundary[i-1:i+2, j-1:j+2]) - 1
                local_curvature = 8 - count
                total_curvature += local_curvature
    return total_curvature

def loss_function(image, segmentation, sigma=3, lambda1=1, lambda2=1, nu=0.1, alpha=0.5, beta=0.5):
    phi = segmentation * 2 - 1
    f1, f2 = compute_f1_f2(image, phi, sigma)
    energy = compute_total_energy(image, segmentation, f1, f2, lambda1, lambda2, nu)
    curvature = compute_curvature_sum(segmentation)
    loss = alpha * energy + beta * curvature
    return loss
