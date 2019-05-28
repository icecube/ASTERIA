from asteria.neutrino import Flavor, Ordering
import unittest

class TestOrdering(unittest.TestCase):
    
    def test_len(self):
        self.assertEqual(len(Ordering), 4)
    
    def test_values(self): 
        self.assertEqual(Ordering.normal.value, 1)
        self.assertEqual(Ordering.inverted.value, 2)    
        self.assertEqual(Ordering.any.value, 3)
        self.assertEqual(Ordering.none.value, 4) 

class TestFlavor(unittest.TestCase):
    
    def test_len_default(self):
        self.assertEqual(len(Flavor), 4)
        
    def test_len_custom(self):
        custom = Flavor({'nu_e' : True})       
        self.assertEqual(len(custom), 1)
        
    def test_values_default(self): 
        self.assertEqual(Flavor.nu_e.value, 1)
        self.assertEqual(Flavor.nu_e_bar.value, -1)    
        self.assertEqual(Flavor.nu_x.value, 2)
        self.assertEqual(Flavor.nu_x_bar.value, -2) 
        
    def test_values_custom(self):
        custom = Flavor({'nu_e' : True})
        self.assertEqual(custom.nu_e.value, 1)
        self.assertEqual(custom.nu_e_bar, -1)    
        self.assertEqual(custom.nu_x, 2)
        self.assertEqual(custom.nu_x_bar, -2)  

    def test_requests_default(self):
        default = {'nu_e'     : True,
                   'nu_e_bar' : True,
                   'nu_x'     : True,
                   'nu_x_bar' : True }

        requests = Flavor.requests
        for (key, val) in default.items():
            with self.subTest(key=key, val=val):
                self.assertIn( key, requests)
                self.assertEqual(requests[key], val)
 
    def test_requests_custom(self):
        custom = {'nu_e'     : True,
                  'nu_e_bar' : False,
                  'nu_x'     : True,
                  'nu_x_bar' : True }

        requests = Flavor(custom).requests
        for (key, val) in custom.items():
            with self.subTest(key=key, val=val):
                self.assertIn( key, requests)
                self.assertEqual(requests[key], val)

    def test_members_default(self):
        self.assertIn(Flavor.nu_e, Flavor)
        self.assertIn(Flavor.nu_e_bar, Flavor)
        self.assertIn(Flavor.nu_x, Flavor)
        self.assertIn(Flavor.nu_x_bar, Flavor)

    def test_members_custom(self):
        custom = {'nu_e'     : True,
                  'nu_e_bar' : False,
                  'nu_x'     : True}

        customFlavor = Flavor(custom)
        self.assertIn(customFlavor.nu_e, customFlavor)
        self.assertIn(customFlavor.nu_x, customFlavor)
        self.assertNotIn(customFlavor.nu_e_bar, customFlavor)
        self.assertNotIn(customFlavor.nu_x_bar, customFlavor)

    def test_errors_no_requests(self):        
        with self.assertRaises(RuntimeError):
            Flavor()

        with self.assertRaises(RuntimeError):
            Flavor({})

        with self.assertRaises(RuntimeError):
            empty = {'nu_e'     : False,
                     'nu_e_bar' : False,
                     'nu_x'     : False,
                     'nu_x_bar' : False }
            Flavor(empty)

        with self.assertRaises(RuntimeError):
            empty = {'nu_e'     : None,
                     'nu_e_bar' : None,
                     'nu_x'     : None,
                     'nu_x_bar' : None }
            Flavor(empty)

    def test_error_invalid_flavors(self):            
        with self.assertRaises(AttributeError):
            custom = {'sterile'  : True,
                      'nu_e_bar' : True,
                      'nu_x'     : False,
                      'nu_x_bar' : False }
            Flavor(custom)

    def test_errors_invalid_requests(self):            
        with self.assertRaises(ValueError):
            custom = {'nu_e'     : 1,
                      'nu_e_bar' : 0,
                      'nu_x'     : 2,
                      'nu_x_bar' : 1. }
            Flavor(custom)
            
    def test_to_tex(self):
        self.assertEqual(Flavor.nu_e.to_tex(), '$\\nu_e$' )
        self.assertEqual(Flavor.nu_e_bar.to_tex(), '$\\overline{\\nu}_e$')
        self.assertEqual(Flavor.nu_x.to_tex(), '$\\nu_x$')
        self.assertEqual(Flavor.nu_x_bar.to_tex(), '$\\overline{\\nu}_x$')
        
    def test_electron(self):
        self.assertTrue(Flavor.nu_e.is_electron)
        self.assertTrue(Flavor.nu_e_bar.is_electron)
        self.assertFalse(Flavor.nu_x.is_electron)
        self.assertFalse(Flavor.nu_x_bar.is_electron)
        
    def test_neutrino(self):
        self.assertTrue(Flavor.nu_e.is_neutrino)
        self.assertFalse(Flavor.nu_e_bar.is_neutrino)
        self.assertTrue(Flavor.nu_x.is_neutrino)
        self.assertFalse(Flavor.nu_x_bar.is_neutrino)
    
    def test_antineutrino(self):
        self.assertFalse(Flavor.nu_e.is_antineutrino)
        self.assertTrue(Flavor.nu_e_bar.is_antineutrino)
        self.assertFalse(Flavor.nu_x.is_antineutrino)
        self.assertTrue(Flavor.nu_x_bar.is_antineutrino)

