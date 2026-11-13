import unittest
import numpy as np
from upi.events import Events

class EventsTest(unittest.TestCase):
    def setUp(self):
        self.events = Events(np.array([0,0,0,1]), np.array([10,20,40,15]), np.array([1.,2.,4.,8.]))
    def test_window_count_and_sum(self):
        np.testing.assert_array_equal(self.events.count(np.array([0]),np.array([20]),-10,20),[3])
        np.testing.assert_allclose(self.events.sum(np.array([0]),np.array([20]),-10,20),[7.])
    def test_values_and_gaps_exclude_current(self):
        np.testing.assert_allclose(self.events.next_value(np.array([0]),np.array([20]),30),[4.])
        np.testing.assert_allclose(self.events.previous_value(np.array([0]),np.array([20]),30),[1.])
        np.testing.assert_allclose(self.events.previous_gap(np.array([0]),np.array([20])),[10.])
        np.testing.assert_allclose(self.events.next_gap(np.array([0]),np.array([20])),[20.])
    def test_unknown_group(self):
        self.assertEqual(self.events.count(np.array([-1]),np.array([20]),-10,10)[0],0)
        self.assertTrue(np.isnan(self.events.next_value(np.array([-1]),np.array([20]),30)[0]))
