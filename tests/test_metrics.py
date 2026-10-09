import unittest
import numpy as np
from tools.collect_metrics import measures,aggregate
from protocols import DATASETS

class MetricsTest(unittest.TestCase):
    def test_separation_direction_and_ties(self):
        a=measures([3,4],[1,2]);self.assertEqual(a['auroc'],100);self.assertEqual(a['fpr95'],0)
        b=measures([-3,-4],[-1,-2],'ood_high');self.assertEqual(a,b)
        tied=measures([1]*20,[1]*10);self.assertEqual(tied['auroc'],50);self.assertEqual(tied['fpr95'],100)
    def test_validation(self):
        for a,b in [([],[1]),([float('nan')],[1]),([[1]],[0])]:
            with self.assertRaises(ValueError):measures(a,b)
        with self.assertRaises(ValueError):measures([1],[0],'auto')
    def test_groups_and_run_sd(self):
        rows=[dict(method='A',backbone='B16',seed=s,dataset=d,fpr95=i+s,auroc=90-i-s,aupr_id=80-i-s)
              for s in (1,3) for i,d in enumerate(DATASETS)]
        expanded,means=aggregate(rows)
        self.assertEqual(len(expanded),20)
        far=next(r for r in means if r['dataset']=='Far')
        self.assertAlmostEqual(far['fpr95'],5.)
        self.assertAlmostEqual(far['fpr95_std'],np.sqrt(2))
        with self.assertRaises(ValueError):aggregate(rows[:-1])
        with self.assertRaises(ValueError):aggregate(rows+[rows[0]])
if __name__=='__main__':unittest.main()
