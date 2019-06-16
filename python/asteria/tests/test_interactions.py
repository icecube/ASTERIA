from asteria.interactions import Interactions
from asteria import interactions
from asteria.neutrino import Flavor
import astropy.units as u
import numpy as np

import unittest

class TestInteractions(unittest.TestCase):
    def test_len_default(self):
        self.assertEqual(len(Interactions), 5)
        
    def test_len_custom(self):
        custom = Interactions({'InvBetaTab' : True})       
        self.assertEqual(len(custom), 1)
        
    def test_values_default(self): 
        self.assertIsInstance(Interactions.InvBetaTab, interactions.InvBetaTab)
        self.assertIsInstance(Interactions.InvBetaPar.value, interactions.InvBetaPar)
        self.assertIsInstance(Interactions.ElectronScatter.value, interactions.ElectronScatter)    
        self.assertIsInstance(Interactions.Oxygen16CC.value, interactions.Oxygen16CC)
        self.assertIsInstance(Interactions.Oxygen16NC.value, interactions.Oxygen16NC) 
        self.assertIsInstance(Interactions.Oxygen18.value, interactions.Oxygen18) 
        
    def test_values_custom(self):
        custom = Interactions({'InvBetaTab' : True})
        self.assertIsInstance(custom.InvBetaTab.value, interactions.InvBetaTab)
        self.assertIsInstance(custom.InvBetaPar, interactions.InvBetaPar)
        self.assertIsInstance(custom.ElectronScatter, interactions.ElectronScatter)    
        self.assertIsInstance(custom.Oxygen16CC, interactions.Oxygen16CC)
        self.assertIsInstance(custom.Oxygen16NC, interactions.Oxygen16NC) 
        self.assertIsInstance(custom.Oxygen18, interactions.Oxygen18)  

    def test_requests_default(self):
        default = {'InvBetaTab'      : False,
                   'InvBetaPar'      : True,
                   'ElectronScatter' : True,
                   'Oxygen16CC'      : True, 
                   'Oxygen16NC'      : True,
                   'Oxygen18'        : True }

        requests = Interactions.requests
        for (key, val) in default.items():
            with self.subTest(key=key, val=val):
                self.assertIn( key, requests)
                self.assertEqual(requests[key], val)
 
    def test_requests_custom(self):
        custom = {'InvBetaTab'      : True,
                  'InvBetaPar'      : False,
                  'ElectronScatter' : False,
                  'Oxygen16CC'      : True, 
                  'Oxygen16NC'      : True,
                  'Oxygen18'        : True }

        requests = Interactions(custom).requests
        for (key, val) in custom.items():
            with self.subTest(key=key, val=val):
                self.assertIn( key, requests)
                self.assertEqual(requests[key], val)
    
    def test_requests_forced_IBD(self):
        custom = {'InvBetaTab' : True,
                  'InvBetaPar' : True}

        requests = Interactions(custom, force = True).requests
        for (key, val) in custom.items():
            with self.subTest(key=key, val=val):
                self.assertIn( key, requests)
                self.assertEqual(requests[key], val)

    def test_members_default(self):
        self.assertIn(Interactions.InvBetaPar, Interactions)
        self.assertIn(Interactions.ElectronScatter, Interactions)
        self.assertIn(Interactions.Oxygen16CC, Interactions)
        self.assertIn(Interactions.Oxygen16NC, Interactions)
        self.assertIn(Interactions.Oxygen18, Interactions)

    def test_members_custom(self):
        custom = {'InvBetaTab'      : True,
                  'InvBetaPar'      : False,
                  'ElectronScatter' : False,
                  'Oxygen16CC'      : True, 
                  'Oxygen16NC'      : True,
                  'Oxygen18'        : True }

        customInteractions = Interactions(custom)
        self.assertIn(customInteractions.InvBetaTab, customInteractions)
        self.assertIn(customInteractions.Oxygen16CC, customInteractions)
        self.assertIn(customInteractions.Oxygen16NC, customInteractions)
        self.assertIn(customInteractions.Oxygen18, customInteractions)
        self.assertNotIn(customInteractions.InvBetaPar, customInteractions)
        self.assertNotIn(customInteractions.ElectronScatter, customInteractions)

    def test_members_forced_IBD(self):
        custom = {'InvBetaTab'      : True,
                  'InvBetaPar'      : True}

        customInteractions = Interactions(custom, force=True)
        self.assertIn(customInteractions.InvBetaTab, customInteractions)
        self.assertIn(customInteractions.InvBetaPar, customInteractions)
    
    def test_errors_empty(self):        
        with self.assertRaises(RuntimeError):
            Interactions()

        with self.assertRaises(RuntimeError):
            Interactions({})

        with self.assertRaises(RuntimeError):
            empty = {'InvBetaTab'      : False,
                     'InvBetaPar'      : False,
                     'ElectronScatter' : False,
                     'Oxygen16CC'      : False, 
                     'Oxygen16NC'      : False,
                     'Oxygen18'        : False }
            Interactions(empty)

        with self.assertRaises(RuntimeError):
            empty = {'InvBetaTab'      : None,
                     'InvBetaPar'      : None,
                     'ElectronScatter' : None,
                     'Oxygen16CC'      : None, 
                     'Oxygen16NC'      : None,
                     'Oxygen18'        : None }
            Interactions(empty)

    def test_errors_Interaction(self):            
        with self.assertRaises(AttributeError):
            custom = {'DoubleBeta'      : True,
                      'InvBetaPar'      : True,
                      'ElectronScatter' : False,
                      'Oxygen16CC'      : True }
            Interactions(custom)

    def test_errors_values(self):            
        with self.assertRaises(ValueError):
            custom = {'InvBetaTab'      : 1,
                      'InvBetaPar'      : 0,
                      'ElectronScatter' : 2,
                      'Oxygen16CC'      : 3., 
                      'Oxygen16NC'      : True,
                      'Oxygen18'        : False }
            Interactions(custom)
    
    def test_errors_IBD(self):            
        with self.assertRaises(RuntimeError):
            custom = {'InvBetaTab' : True,
                      'InvBetaPar' : True}
            Interactions(custom)    
