"""Isaac Sim 6.0 adapter. Import only after creating SimulationApp."""
import numpy as np
from common.contract import JOINTS, FPS, RESOLUTION, TARGET, CAMERA_EYE, CAMERA_TARGET, vector


def array(x):
    return np.asarray(x.numpy() if hasattr(x, 'numpy') else x)


class FrankaScene:
    def __init__(self, app):
        import omni.usd
        import omni.replicator.core as rep
        import isaacsim.core.experimental.utils.app as app_utils
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
        from pxr import Gf, UsdGeom, UsdPhysics, UsdLux
        app_utils.enable_extension('isaacsim.robot.experimental.manipulators.examples')
        from isaacsim.robot.experimental.manipulators.examples.franka import FrankaPickPlace
        from isaacsim.core.experimental.prims import RigidPrim

        self.app, self.sm, self.rm = app, SimulationManager, RenderingManager
        self.app_utils = app_utils
        stage = omni.usd.get_context().get_stage()

        def box(path, pos, scale, color):
            b = UsdGeom.Cube.Define(stage, path)
            b.CreateSizeAttr(1.)
            b.AddTranslateOp().Set(Gf.Vec3d(*pos))
            b.AddScaleOp().Set(Gf.Vec3d(*scale))
            b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(b.GetPrim())

        box('/World/Table', [0.3, 0, -0.025], [1.4, 1.2, 0.05], [0.55, 0.42, 0.30])
        box('/World/ground_plane', [0, 0, -0.825], [10, 10, 0.05], [0.18]*3)
        for i, (x, y) in enumerate([(-.3, -.5), (-.3, .5), (.9, -.5), (.9, .5)]):
            box(f'/World/Leg{i}', [x, y, -.425], [.06, .06, .75], [.25]*3)
        UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.)
        # Called at 20 Hz; deliberately slower than the original demo for data collection.
        # Official target_position is the hand-link target, not the cube center.
        # Its approach phase uses a 0.10 m hand-to-cube offset.
        self.expert = FrankaPickPlace(events_dt=[80, 60, 30, 60, 100, 30, 30])
        self.expert.setup_scene(cube_initial_position=[.45, -.2, .0258],
                               cube_size=[.05]*3, target_position=np.asarray(TARGET) + [0., 0., .10])
        UsdGeom.Gprim(stage.GetPrimAtPath('/World/Cube')).CreateDisplayColorAttr([Gf.Vec3f(.9, .1, .1)])
        self.robot = self.expert.robot
        self.cube = RigidPrim('/World/Cube')

        cam = UsdGeom.Camera.Define(stage, '/World/ObservationCamera')
        cam.CreateFocalLengthAttr(24.)
        cam.CreateHorizontalApertureAttr(32.)
        cam.CreateVerticalApertureAttr(32.)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
        transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*CAMERA_EYE),
                                         Gf.Vec3d(*CAMERA_TARGET), Gf.Vec3d(0, 0, 1)).GetInverse()
        cam.AddTransformOp().Set(transform)
        self.product = rep.create.render_product(str(cam.GetPath()), RESOLUTION)
        self.rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        self.rgb.attach([self.product])
        # A headless run may not have an active viewport; the observation
        # render product above remains independent of this display-only view.
        try:
            viewport_camera = ViewportManager.get_camera()
            if viewport_camera is not None:
                ViewportManager.set_camera_view(camera=viewport_camera,
                                                eye=[1.4, -1.3, 1.2], target=[.3, 0, .2])
        except (RuntimeError, AssertionError) as exc:
            print(f'Viewport unavailable (observation camera still configured): {exc}')
        SimulationManager.setup_simulation(dt=1/60, device='cpu')
        RenderingManager.set_dt(1/60)
        SimulationManager.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
        app_utils.play()
        app_utils.update_app(steps=20)
        self.expert.reset()
        names = list(self.robot.dof_names)
        if set(names) != set(JOINTS):
            raise RuntimeError(f'Unexpected DOFs: {names}')
        self.indices = [names.index(j) for j in JOINTS]
        self.home = self.state()
        low, high = self.robot.get_dof_limits()
        self.lower = array(low)[0, self.indices]
        self.upper = array(high)[0, self.indices]
        for _ in range(5):
            self.rm.render()
        self.reset([.45, -.2, .0258])

    def state(self):
        return vector(array(self.robot.get_dof_positions())[0, self.indices])

    def command(self):
        return vector(array(self.robot.get_dof_position_targets())[0, self.indices])

    def apply(self, target):
        target = vector(target)
        clipped = np.clip(target, self.lower, self.upper)
        self.robot.set_dof_position_targets(clipped, dof_indices=self.indices)
        return clipped, bool(np.any(clipped != target))

    def reset(self, initial):
        self.expert.reset()
        self.robot.set_dof_positions(self.home, dof_indices=self.indices)
        self.robot.set_dof_velocities(np.zeros(9), dof_indices=self.indices)
        self.apply(self.home)
        self.cube.set_world_poses(positions=np.asarray(initial)[None], orientations=[[1., 0., 0., 0.]])
        self.cube.set_velocities(linear_velocities=[[0., 0., 0.]],
                                 angular_velocities=[[0., 0., 0.]])
        for _ in range(10):
            self.advance()
        # No expert actions have been issued during settling.
        self.ticks = 0
        self.start_time = self.sm.get_simulation_time()

    def cube_position(self):
        return array(self.cube.get_world_poses()[0])[0].copy()

    def observation(self):
        # Render without advancing physics. State and image describe the same instant.
        self.rm.render()
        self.rm.render()
        image = np.asarray(self.rgb.get_data())
        if image.shape[:2] != RESOLUTION[::-1] or image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(f'RGB annotator not ready: {image.shape}')
        return self.state(), np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)

    def advance(self):
        if not self.app.is_running():
            raise KeyboardInterrupt('Simulator window closed')
        before = self.sm.get_simulation_time()
        self.sm.step(steps=3)
        self.rm.render()
        after = self.sm.get_simulation_time()
        if not np.isclose(after-before, 1/FPS, atol=1e-4):
            raise RuntimeError(f'Unexpected physics increment {after-before}; expected {1/FPS}. '
                               'Check Isaac Sim 6.0 manual stepping; do not record mis-timed data.')

    def close(self):
        self.rgb.detach([self.product])
        self.product.destroy()
        self.app_utils.stop()
