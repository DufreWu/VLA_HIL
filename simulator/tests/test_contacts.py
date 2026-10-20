import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from simulator.evaluation.contacts import ContactMonitor, FINGERS


class ContactMonitorTests(unittest.TestCase):
    def test_reports_track_contacts_and_threshold(self):
        monitor = ContactMonitor(1.)
        header = lambda a, b, n=1: SimpleNamespace(actor0=a, actor1=b,
                                   contact_data_offset=0, num_contact_data=n)
        pxr = SimpleNamespace(PhysicsSchemaTools=SimpleNamespace(intToSdfPath=lambda p: p))
        with patch.dict(sys.modules, {'pxr': pxr}):
            for finger in FINGERS:
                monitor._on_contacts([header('/World/Cube', '/World/robot/'+finger)],
                                     [SimpleNamespace(impulse=[.001, 0, 0])])
            self.assertEqual(monitor.fingers, FINGERS)
            self.assertFalse(monitor.collision)
            monitor._on_contacts([header('/World/robot/panda_hand', '/World/Table')],
                                 [SimpleNamespace(impulse=[.01, 0, 0])])
            self.assertFalse(monitor.collision)
            monitor._on_contacts([header('/World/robot/panda_hand', '/World/Table')],
                                 [SimpleNamespace(impulse=[.02, 0, 0])])
            self.assertTrue(monitor.collision)
            self.assertAlmostEqual(monitor.events[0]['force_n'], 1.2)
            monitor.clear()
            monitor._on_contacts([header('/World/Cube', '/World/Table', 0)], [])
            self.assertFalse(monitor.collision)
            self.assertEqual(monitor.fingers, set())
