"""Contact classification independent of the PhysX Python runtime."""
ROBOT = '/World/robot/'
CUBE = '/World/Cube'
FINGERS = {'panda_leftfinger', 'panda_rightfinger'}


def is_finger(path):
    return path.startswith(ROBOT) and path.rsplit('/', 1)[-1] in FINGERS


def unintended(a, b):
    if b.startswith(ROBOT):
        a, b = b, a
    if a.startswith(ROBOT):
        if b.startswith(ROBOT):
            return False  # Asset self-collision filters define this separately.
        if b == CUBE and is_finger(a):
            return False
        if a.rsplit('/', 1)[-1] == 'panda_link0' and b == '/World/Table':
            return False
        return True
    if a == CUBE or b == CUBE:
        other = b if a == CUBE else a
        return other != '/World/Table'
    return False


class ContactMonitor:
    """Subscribe before physics starts; accumulate each physics step's reports."""
    def __init__(self, force_threshold):
        self.threshold = force_threshold
        self.subscription = None
        self.clear()

    def clear(self):
        self.collision = False
        self.fingers = set()
        self.events = []
        self.report_count = 0

    def install(self, stage):
        from pxr import PhysxSchema, UsdPhysics
        from omni.physx import get_physx_simulation_interface
        count = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if (path.startswith(ROBOT) or path == CUBE) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
                count += 1
        if count < 3:
            raise RuntimeError('Robot/cube rigid bodies missing; cannot install contact monitoring')
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contacts)

    def _on_contacts(self, headers, data):
        from pxr import PhysicsSchemaTools
        for header in headers:
            if not header.num_contact_data:
                continue
            self.report_count += 1
            a, b = (str(PhysicsSchemaTools.intToSdfPath(x)) for x in (header.actor0, header.actor1))
            samples = [data[i] for i in range(header.contact_data_offset,
                                             header.contact_data_offset + header.num_contact_data)]
            # Sum magnitudes avoids cancelling opposing impulses; physics dt is 1/60.
            force = sum(sum(float(x)**2 for x in s.impulse)**0.5 for s in samples)*60.0
            if force <= 0:
                continue
            if a == CUBE and is_finger(b):
                self.fingers.add(b.rsplit('/', 1)[-1])
            if b == CUBE and is_finger(a):
                self.fingers.add(a.rsplit('/', 1)[-1])
            if force >= self.threshold and unintended(a, b):
                self.collision = True
                self.events.append(dict(actor0=a, actor1=b, force_n=force))

    def close(self):
        self.subscription = None
