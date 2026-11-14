import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import specialist_sweep as tool


class SpecialistTests(unittest.TestCase):
    def test_views_are_declared_and_exclude_only_named_features(self):
        X = pd.DataFrame({'amount':[1.], 'band':[2], 'snd_band_1_3h':[1], 'dev_ext_legit':[0],
                          'ec_sender_peers':[2], 'ch_len':[3]})
        self.assertNotIn('ec_sender_peers',tool.feature_view(X,'base'))
        self.assertEqual(list(tool.feature_view(X,'without_bands')),['amount','ec_sender_peers','ch_len'])
        pd.testing.assert_frame_equal(tool.feature_view(X,'events'),X)
        with self.assertRaises(ValueError): tool.feature_view(X,'unknown')

    def test_binary_scores_normalize_and_reject_invalid_values(self):
        p = tool.specialist_probabilities(np.array([[0.]*11,[.2]*11]))
        np.testing.assert_allclose(p.sum(1),1)
        self.assertEqual(p[0,0],1)
        for q in (np.ones((2,12)),np.full((2,11),np.nan),np.full((2,11),1.1)):
            with self.assertRaises(ValueError): tool.specialist_probabilities(q)

    def test_ovr_chronology_refit_rounds_and_checkpoint_resume(self):
        import lightgbm
        calls=[]
        class Mock:
            classes_=np.array([0,1])
            best_iteration_=7
            def __init__(self,**kwargs): self.params=kwargs; self.n_estimators=kwargs['n_estimators']; self.booster_=self
            def fit(self,X,y,**kwargs): calls.append((self.params,kwargs,len(X))); return self
            def predict_proba(self,X): return np.tile([.98,.02],(len(X),1))
            def save_model(self,path): Path(path).write_text('mock')
        rows=pd.DataFrame({'txn_id':[f't{i}' for i in range(36)],'y':np.tile(np.arange(12),3),
            'weight':1.,'fit':np.arange(36)<12,'query':np.arange(36)>=12,
            'early_stop':(np.arange(36)>=12)&(np.arange(36)<24)})
        X=pd.DataFrame({'x':np.arange(36)})
        with tempfile.TemporaryDirectory() as temp, patch.object(lightgbm,'LGBMClassifier',Mock):
            folder=Path(temp)
            p,info=tool.fit_predict(X,rows,{'categorical_mappings':{}},tool.SPECS['ovr_base'],folder)
            self.assertEqual(p.shape,(24,12)); self.assertEqual(info['rounds'],[7]*11)
            self.assertEqual(len(calls),11)
            self.assertTrue(all(n==12 and len(kw['eval_set'][0][0])==12 for _,kw,n in calls))
            tool.fit_predict(X,rows,{'categorical_mappings':{}},tool.SPECS['ovr_base'],folder)
            self.assertEqual(len(calls),11)
            refit=folder/'refit'; refit.mkdir()
            tool.fit_predict(X,rows,{'categorical_mappings':{}},tool.SPECS['ovr_base'],refit,[7]*11)
            self.assertTrue(all('eval_set' not in kw and params['n_estimators']==7 for params,kw,n in calls[11:]))
            (folder/'class_1.txt').write_text('changed')
            with self.assertRaises(ValueError):
                tool.fit_predict(X,rows,{'categorical_mappings':{}},tool.SPECS['ovr_base'],folder)

    def test_linear_preprocessing_fits_only_training_rows(self):
        import sklearn.linear_model
        captured=[]
        class Mock:
            classes_=np.arange(12)
            n_iter_=np.array([3])
            def __init__(self,**kwargs): pass
            def fit(self,X,y,**kwargs): captured.append((X.shape,y,kwargs)); return self
            def predict_proba(self,X): return np.full((X.shape[0],12),1/12)
        row=pd.DataFrame({'y':np.tile(np.arange(12),2),'weight':1.,'fit':np.arange(24)<12,
            'query':np.arange(24)>=12,'early_stop':np.arange(24)>=12})
        X=pd.DataFrame({'amount':np.arange(24,dtype=float),'cat':[0]*12+[9]*12})
        with tempfile.TemporaryDirectory() as temp, patch.object(sklearn.linear_model,'LogisticRegression',Mock),patch('joblib.dump') as dump:
            p,info=tool.fit_predict(X,row,{'categorical_mappings':{'cat':['a']}},tool.SPECS['linear_base'],Path(temp))
            transform=dump.call_args.args[0]['transform']
            np.testing.assert_array_equal(transform.named_transformers_['categorical'].categories_[0],[0])
            self.assertEqual(captured[0][0][0],12)
            self.assertEqual(p.shape,(12,12)); self.assertTrue(info['converged'])


if __name__=='__main__': unittest.main()
