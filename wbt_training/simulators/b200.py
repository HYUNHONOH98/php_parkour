"""PHP clock, B200 patch capacity, and explicit native material commands."""
import holosoma.simulator.isaacsim.isaacsim as core
from wbt_training.simulators.isaacsim import IsaacSim as PhpIsaacSim


def install_preview_command():
    """Keep native USD authoring while avoiding missing/ambiguous short names."""
    import omni.kit.commands as commands
    from omni.usd.commands.usd_commands import CreatePreviewSurfaceMaterialPrimCommand
    native_class = CreatePreviewSurfaceMaterialPrimCommand
    qualified = native_class.__module__ + '.CreatePreviewSurfaceMaterialPrim'
    commands.register(native_class)
    assert commands.get_command_class(qualified) is native_class
    original = commands.execute
    if not getattr(original, '_php_preview_command_guard', False):
        def execute(name, *args, **kwargs):
            if name == 'CreatePreviewSurfaceMaterialPrim':
                commands.register(native_class)
                return original(qualified, *args, **kwargs)
            return original(name, *args, **kwargs)
        execute._php_preview_command_guard = True
        commands.execute = execute
    return qualified


class IsaacSim(PhpIsaacSim):
    def __init__(self, tyro_config, terrain_manager, device):
        num_envs = tyro_config.training.num_envs
        original = core.PhysxCfg
        headless = tyro_config.training.headless
        if headless:
            self._php_preview_surface_command_name = install_preview_command()

        def capacity(*args, **kwargs):
            kwargs['gpu_max_rigid_patch_count'] = max(kwargs.get('gpu_max_rigid_patch_count', 10 * 2**15), 32 * num_envs)
            return original(*args, **kwargs)

        core.PhysxCfg = capacity
        try:
            super().__init__(tyro_config, terrain_manager, device)
        finally:
            core.PhysxCfg = original
