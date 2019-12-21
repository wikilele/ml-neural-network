import unittest
from datasets.dataset import *

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.ds = MonkDataset.load('tests/assets/monks.test')
    
    def test_size(self):

        assert self.ds.size() == 6

    def test_batch1(self):

        for b in self.ds.batch(2):
            assert len(b) == 2
    
    def test_batch2(self):

        for b in self.ds.batch(3):
            assert len(b) == 3 or len(b) == 1
    
    def test_splitting(self):
        train, val = self.ds.split()
        assert train.size() == 4
        assert val.size() == 2
