"""
Custom hook to freeze specific modules during training
"""
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FreezeWorldModelHook(Hook):
    """
    Hook to freeze World Model parameters during training.
    
    This is useful for two-stage training where:
    1. Pre-train World Model
    2. Freeze WM and train planning with WM supervision
    
    Args:
        freeze_modules (list): List of module names to freeze
    """
    
    def __init__(self, freeze_modules=None):
        self.freeze_modules = freeze_modules or ['world_model']
    
    def before_run(self, runner):
        """Freeze specified modules before training starts"""
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        for module_name in self.freeze_modules:
            self._freeze_module(model, module_name)
    
    def _freeze_module(self, model, module_name):
        """Recursively freeze a module by name"""
        # Navigate to the module
        parts = module_name.split('.')
        current = model
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                print(f"Warning: Module {module_name} not found in model")
                return
        
        # Freeze all parameters
        frozen_count = 0
        for param in current.parameters():
            param.requires_grad = False
            frozen_count += 1
        
        print(f"Frozen {frozen_count} parameters in {module_name}")
        
        # Set to eval mode
        current.eval()
        
        # Register hook to keep it in eval mode
        def set_eval(module, input):
            module.eval()
        
        current.register_forward_pre_hook(set_eval)


@HOOKS.register_module()
class LossWeightScheduleHook(Hook):
    """
    Hook to schedule loss weights during training.
    
    Useful for curriculum learning or focusing on specific losses.
    
    Args:
        loss_weights (dict): Mapping of loss names to weights
        schedule (str): 'constant', 'linear', 'cosine'
        start_iter (int): When to start applying weights
        end_iter (int): When to reach final weights
    """
    
    def __init__(self, 
                 loss_weights=None,
                 schedule='constant',
                 start_iter=0,
                 end_iter=None):
        self.loss_weights = loss_weights or {}
        self.schedule = schedule
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.initial_weights = {}
    
    def before_run(self, runner):
        """Store initial loss weights"""
        # This would require access to loss computation
        # For now, we just store target weights
        pass
    
    def before_train_iter(self, runner):
        """Adjust loss weights based on iteration"""
        if self.schedule == 'constant':
            # Already at target weights
            pass
        elif self.schedule == 'linear':
            # Linear interpolation
            if runner.iter < self.start_iter:
                alpha = 0.0
            elif self.end_iter and runner.iter > self.end_iter:
                alpha = 1.0
            else:
                alpha = (runner.iter - self.start_iter) / (self.end_iter - self.start_iter)
            
            # Update weights (this is a placeholder - actual implementation
            # would need to modify the model's loss computation)
            runner.logger.info(f"Loss weight schedule: alpha={alpha:.3f}")

