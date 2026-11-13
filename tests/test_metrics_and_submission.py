import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from upi.artifacts import validate_submission, write_submission
from upi.metrics import apply_weights, classification_report_dict, fit_class_weights

class MetricsSubmissionTest(unittest.TestCase):
    def test_report_has_all_classes(self):
        report=classification_report_dict(np.array([0,1,1]),np.array([0,1,0]))
        self.assertEqual(len(report["per_class"]),12)
    def test_weight_fit(self):
        p=np.full((6,12),.001); p[:,0]=.7; p[:3,1]=.4; y=np.array([1,1,1,0,0,0])
        w=fit_class_weights(p,y,grid=np.array([1.,2.,4.]),sweeps=1)
        self.assertEqual(apply_weights(p,w).shape,y.shape); self.assertEqual(w[0],1.)
    def test_submission_order(self):
        sample=pd.DataFrame({"id":["b","a"],"label":["legit","legit"]})
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"s.csv"
            write_submission(path,np.array(["a","b"]),np.array([1,0]),sample)
            result=pd.read_csv(path)
            self.assertEqual(result.id.tolist(),["b","a"])
            self.assertEqual(result.label.tolist(),["legit","mule_account"])
            validate_submission(result,sample)
    def test_invalid_label(self):
        sample=pd.DataFrame({"id":["a"],"label":["legit"]})
        with self.assertRaises(ValueError):
            validate_submission(pd.DataFrame({"id":["a"],"label":["bad"]}),sample)
