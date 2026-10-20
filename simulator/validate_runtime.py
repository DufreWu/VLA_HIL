"""Isaac Sim integration probe: camera timing and deliberate robot/table contact.

Run in the Isaac Sim environment. This is a destructive *simulation* probe,
separate from benchmark episodes; it deliberately commands the gripper into the table.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/contact_probe.json'))
    args = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    from simulator.evaluation.contacts import ContactMonitor
    from simulator.franka.scene import FrankaScene
    monitor = ContactMonitor(1.)
    scene = None
    result = {'success': False}
    exit_code = 0
    try:
        scene = FrankaScene(app, before_simulation=monitor.install)
        before = scene.sm.get_simulation_time()
        state, rgb = scene.observation()
        after = scene.sm.get_simulation_time()
        assert abs(before-after) < 1e-7, 'Capture advanced physics'
        assert rgb.shape == (256, 256, 3), rgb.shape
        events = []
        for tick in range(160):
            # Descend away from the cube, deliberately crossing the tabletop.
            height = .30 if tick < 40 else max(.03, .30-(tick-40)*.0045)
            scene.robot.set_end_effector_pose(position=[.50, .20, height],
                         orientation=scene.robot.get_downward_orientation())
            scene.robot.close_gripper()
            scene.apply(scene.command())
            monitor.clear()
            scene.advance()
            events.extend(monitor.events)
            if events:
                break
        assert events, 'Deliberate table contact was not detected'
        result = dict(success=True, camera_shape=list(rgb.shape),
                      capture_physics_time_change_s=after-before,
                      collision_threshold_n=1., detected_contacts=events)
    except BaseException as exc:
        exit_code = 1
        import traceback
        traceback.print_exc()
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        monitor.close()
        try:
            if scene is not None:
                scene.close()
        finally:
            app.close(exit_code=exit_code)


if __name__ == '__main__':
    main()
