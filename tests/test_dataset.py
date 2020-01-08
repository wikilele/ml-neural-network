import unittest
from datasets.dataset import *
import numpy as np

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
        train, val = self.ds.split(2/3)
        assert train.size() == 4
        assert val.size() == 2


    def test_normalize(self):
        ds = Dataset(np.array([ [1,[-2,1,2],[1]] , [2,[1,1.5,1],[1.5]], [3,[4,3,0],[1]] ]))
        ds.normalize()
        ds.print()