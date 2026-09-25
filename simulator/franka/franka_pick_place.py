"""Isaac Sim 6.0: Franka pick/place using the experimental example API.
Run: ./python.sh /absolute/path/franka_pick_place.py
Optional: --headless --test
Based on the 6.0 Adding a Manipulator Robot API documentation.
Only syntax checked here; local simulator validation is required.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--headless', action='store_true')
parser.add_argument('--test', action='store_true', help='Exit after the sequence, with a frame timeout')
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
app = SimulationApp({'headless': args.headless})


def main():
    import omni.usd
    import isaacsim.core.experimental.utils.app as app_utils
    from pxr import Gf, UsdGeom, UsdPhysics, UsdLux

    extension = 'isaacsim.robot.experimental.manipulators.examples'
    app_utils.enable_extension(extension)
    try:
        from isaacsim.robot.experimental.manipulators.examples.franka import FrankaPickPlace
    except ImportError as exc:
        raise RuntimeError(
            'This script targets the Isaac Sim 6.0 FrankaPickPlace API. '
            'Check your exact 6.x version and installed example extensions.'
        ) from exc
    from isaacsim.core.simulation_manager import SimulationManager

    stage = omni.usd.get_context().get_stage()

    def fixed_box(path, center, dimensions, color):
        box = UsdGeom.Cube.Define(stage, path)
        box.CreateSizeAttr(1.0)
        box.AddTranslateOp().Set(Gf.Vec3d(*center))
        box.AddScaleOp().Set(Gf.Vec3d(*dimensions))
        box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        # Collider without RigidBodyAPI: static support geometry.
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())

    # Work surface at z=0 keeps the controller's original robot frame.
    fixed_box('/World/Table', (0.3, 0, -0.025), (1.4, 1.2, 0.05), (0.55, 0.42, 0.30))
    for index, (x, y) in enumerate([(-0.3, -0.5), (-0.3, 0.5), (0.9, -0.5), (0.9, 0.5)]):
        fixed_box(f'/World/Leg_{index}', (x, y, -0.425), (0.06, 0.06, 0.75), (0.25, 0.25, 0.25))
    fixed_box('/World/Floor', (0, 0, -0.825), (10, 10, 0.05), (0.18, 0.18, 0.18))
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(1000.0)

    controller = FrankaPickPlace(events_dt=[80, 60, 30, 60, 100, 30, 30])
    controller.setup_scene(
        cube_initial_position=[0.45, -0.20, 0.0258],
        cube_size=[0.05, 0.05, 0.05],
        target_position=[0.45, 0.20, 0.0258],
    )
    cube_prim = stage.GetPrimAtPath('/World/Cube')
    if cube_prim.IsValid():
        UsdGeom.Gprim(cube_prim).CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.1, 0.1)])

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device='cpu')
    SimulationManager.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
    app_utils.play()
    app_utils.update_app(steps=20)
    controller.reset()

    frames = 0
    settled = 0
    announced = False
    budget = int(sum(controller.events_dt)) + 300
    print('Isaac Sim 6.0 Franka demo: pick, transfer, release.')
    print('Close the window to exit; rerun the script for a fresh episode.')
    while app.is_running():
        app.update()
        if not app_utils.is_playing():
            continue
        frames += 1
        if not controller.is_done():
            controller.forward()
        else:
            if not announced:
                print('Controller sequence finished; this does not certify grasp success.')
                announced = True
            # Give the released cube time to settle before exiting a test run.
            settled += 1
            if args.test and settled >= 60:
                break
        if args.test and frames >= budget:
            raise RuntimeError('Controller exceeded the frame budget.')
    app_utils.stop()


try:
    main()
finally:
    app.close()
