import torch
import torch.nn as nn
import numpy as np
from odl.contrib.torch import OperatorModule

class PhysicsOperator:
    def __init__(self, geometry_config):
        self.geometry = self.setup_geometry(geometry_config)
        self.forward_op = OperatorModule(self.create_forward_operator())
        self.backward_op = OperatorModule(self.create_backprojection_operator())
        
    def setup_geometry(self, config):
        # Implement ODL geometry setup
        import odl
        space = odl.uniform_discr([-128, -128], [128, 128], [512, 512], dtype='float32')
        angle_partition = odl.uniform_partition(0, 2*np.pi, config['angles'])
        detector_partition = odl.uniform_partition(-360, 360, config['det_width'])
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        return geometry
    
    def create_forward_operator(self):
        import odl
        return odl.tomo.RayTransform(self.geometry.domain, self.geometry)
    
    def create_backprojection_operator(self):
        return self.create_forward_operator().adjoint
    
    def fbp(self, sinogram, filter_type="hann"):
        import odl
        fbp_op = odl.tomo.fbp_op(self.create_forward_operator(), filter_type=filter_type)
        return OperatorModule(fbp_op)(sinogram)
    
    def poisson_thinning(self, clean_sinogram, dose_level):
        """Apply Poisson thinning to simulate dose reduction"""
        # Convert to photons (assuming clean_sinogram is in attenuation units)
        I0 = 10000  # Initial photon count
        attenuated_photons = I0 * torch.exp(-clean_sinogram)
        
        # Apply dose reduction
        scaled_photons = attenuated_photons * dose_level
        
        # Add Poisson noise
        noisy_photons = torch.poisson(scaled_photons)
        
        # Convert back to attenuation units
        noisy_sinogram = -torch.log(noisy_photons / (I0 * dose_level + 1e-10))
        
        return noisy_sinogram