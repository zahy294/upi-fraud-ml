"""Resumable six-candidate chronological sweep. Training is user initiated."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from event_boost_train import preflight, append_features
from hierarchical_train import read, write, prepare, verify_complete
from src.upi.common import LABELS
from src.upi.fit_submit import file_hash, ids_hash, align_proba, save_probability_artifact, load_probability_artifact
from src.upi.make_submission import (prepare_rule_context, slice_rule_context, select_member_policy,
    predict_labels, evaluate_predictions, write_submission)

SPECS = {f"{model}_{view}": {"model": model, "view": view, "seed": 642, "threads": 4,
    "rounds": 600, "patience": 75, "linear_iterations": 200}
    for model in ("ovr", "linear") for view in ("base", "events", "without_bands")}


def feature_view(X, view):
    if view == "base":
        columns = [c for c in X if not c.startswith("ec_")]
    elif view == "events":
        columns = list(X)
    elif view == "without_bands":
        columns = [c for c in X if "band" not in c.lower() and c != "dev_ext_legit"]
    else:
        raise ValueError("unknown feature view")
    return X.loc[:, columns]


def specialist_probabilities(binary):
    q = np.asarray(binary, float)
    if q.ndim != 2 or q.shape[1] != 11 or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("eleven finite binary probabilities required")
    # Independent detectors are not a joint distribution. This explicit
    # normalization creates decision scores; held-out offsets calibrate them.
    p = np.column_stack([np.maximum(1 - q.sum(axis=1), 1e-9), q])
    return p / p.sum(axis=1, keepdims=True)


def fit_predict(X, rows, schema, spec, folder, rounds=None):
    fit, query = rows['fit'].to_numpy(), rows['query'].to_numpy()
    stop = rows.early_stop.to_numpy() if rounds is None else np.zeros(len(rows), bool)
    y, w = rows.y.to_numpy(dtype=int), rows.weight.to_numpy()
    if not np.array_equal(np.unique(y[fit]), np.arange(12)):
        raise ValueError("all twelve fitting classes required")
    cats = [c for c in schema['categorical_mappings'] if c in X]
    if spec['model'] == 'ovr':
        import lightgbm as lgb
        output, iterations = [], []
        for k in range(1, 12):
            checkpoint = folder / f'class_{k}.npz'
            info_path = folder / f'class_{k}.json'
            model_path = folder / f'class_{k}.txt'
            if info_path.exists():
                info = read(info_path)
                if file_hash(checkpoint) != info['probability_hash'] or file_hash(model_path) != info['model_hash']:
                    raise ValueError('specialist checkpoint changed')
                with np.load(checkpoint, allow_pickle=False) as saved:
                    if not np.array_equal(saved['ids'], rows.loc[query, 'txn_id'].to_numpy(dtype=str)):
                        raise ValueError('specialist checkpoint IDs differ')
                    p = saved['p']
                best = info['rounds']
            else:
                print(f"  Specialist {k}/11: {LABELS[k]}", flush=True)
                model = lgb.LGBMClassifier(objective='binary', metric='average_precision',
                    n_estimators=spec['rounds'] if rounds is None else rounds[k-1],
                    learning_rate=.04, num_leaves=31, min_child_samples=40,
                    colsample_bytree=.75, subsample=.8, subsample_freq=1, reg_lambda=5.,
                    random_state=spec['seed']+k, n_jobs=spec['threads'], deterministic=True, verbosity=-1)
                kwargs = {'sample_weight': w[fit], 'categorical_feature': cats}
                if rounds is None:
                    if len(np.unique(y[stop] == k)) != 2:
                        raise ValueError('both stopping classes required')
                    kwargs.update(eval_set=[(X.loc[stop], (y[stop] == k).astype(int))],
                        callbacks=[lgb.early_stopping(spec['patience'], first_metric_only=True, verbose=False),
                                   lgb.log_evaluation(100)])
                model.fit(X.loc[fit], (y[fit] == k).astype(int), **kwargs)
                if not np.array_equal(model.classes_, [0, 1]):
                    raise ValueError('unexpected binary class order')
                p = model.predict_proba(X.loc[query])[:, 1]
                best = int(model.best_iteration_ or model.n_estimators)
                model.booster_.save_model(str(model_path))
                np.savez_compressed(checkpoint, ids=rows.loc[query,'txn_id'].to_numpy(dtype=str), p=p)
                write(info_path, {'rounds': best, 'probability_hash': file_hash(checkpoint),
                                  'model_hash': file_hash(model_path)})
                del model
            output.append(p)
            iterations.append(best)
        return specialist_probabilities(np.column_stack(output)), {'rounds': iterations}
    if spec['model'] != 'linear':
        raise ValueError('unknown learner')
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import joblib
    numeric = [c for c in X if c not in cats]
    transform = ColumnTransformer([
        ('numeric', make_pipeline(FunctionTransformer(signed_log),
            SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
            StandardScaler()), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cats)], sparse_threshold=1.)
    matrix = transform.fit_transform(X.loc[fit])
    model = LogisticRegression(solver='saga', penalty='l2', C=1., max_iter=spec['linear_iterations'],
                               tol=.001, random_state=spec['seed'])
    # Normalize weights for optimizer stability, retaining inverse-sampling ratios.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(matrix, y[fit], sample_weight=w[fit]/w[fit].mean())
    messages = [str(item.message) for item in caught]
    for message in messages:
        print(f'Linear solver warning: {message}', flush=True)
    del matrix
    p = align_proba(model.predict_proba(transform.transform(X.loc[query])), model.classes_)
    joblib.dump({'transform': transform, 'model': model, 'columns': list(X)}, folder/'model.joblib', compress=3)
    return p, {'rounds': [], 'iterations': model.n_iter_.tolist(), 'warnings': messages,
               'converged': not any(issubclass(item.category, ConvergenceWarning) for item in caught)}


def signed_log(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.where(np.isfinite(values), values, np.nan)
    return np.sign(values) * np.log1p(np.abs(values))


def context_from_cache(cache, rows):
    query = rows['query'].to_numpy()
    qrows = rows.loc[query].reset_index(drop=True)
    context = prepare_rule_context(qrows[['txn_id','timestamp','receiver_type']],
        pd.read_parquet(cache/'v3.parquet').loc[query].reset_index(drop=True),
        pd.read_parquet(cache/'v5.parquet').loc[query].reset_index(drop=True))
    return qrows, np.load(cache/'groups.npy', allow_pickle=False)[query], context


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['check','validate','refit'], default='check')
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--output-dir', type=Path, default=Path('runs/specialists_v12'))
    a = ap.parse_args(argv)
    root = a.root.resolve()
    out = (root/a.output_dir).resolve()
    source, features, teacher, reference = [root/'runs'/s for s in
        ('hierarchy_v9','forest_v10','challenge_v8','fraud_gate_comparison')]
    if out == root or root not in out.parents or any(out == p or p in out.parents or out in p.parents
                                                   for p in (source,features,teacher,reference)):
        raise ValueError('output must be a separate folder inside the project')
    sm, inputs, base_manifest = preflight(root,source,features,teacher,reference)
    manifest = {'source': base_manifest, 'specs': SPECS, 'code': file_hash(Path(__file__)),
                'sklearn': importlib.metadata.version('scikit-learn'),
                'reference_code': {n:file_hash(root/n) for n in ('compare_saved_models.py','compare_diverse_models.py','postprocess_saved.py')},
                'reference_artifacts': {f'{stage}/{n}':file_hash(root/'runs/local_podium/ensemble'/stage/n)
                    for stage in ('validation','refit') for n in ('manifest.json','A_proba.npz','B_proba.npz','C_proba.npz','meta.parquet','features_v3.parquet','features_v5.parquet','groups.npy')}}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['signature'] = signature
    if a.stage == 'check':
        print('Preflight passed. Six candidates; three sets of eleven binary detectors plus three linear models. No training.')
        return 0
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists() and read(out/'manifest.json') != manifest:
        raise ValueError('source/config/environment changed; choose a fresh output folder')
    write(out/'manifest.json',manifest)
    validation = a.stage == 'validate'
    stage = 'validation' if validation else 'refit'
    if not validation:
        verify_complete(out/'validation',signature)
        report = read(out/'validation/report.json')
        if not report['eligible_for_refit']:
            raise ValueError('no corroborated improvement; refit disabled')
        schema_dir = out/'validation/cache'
        schema_dir.mkdir(exist_ok=True)
        (schema_dir/'schema.json').write_bytes((source/'validation/cache/schema.json').read_bytes())
        prepare(root,teacher,out,sm['config'],inputs,'refit')
    cache = source/'validation/cache' if validation else out/'refit/cache'
    rows = pd.read_parquet(cache/'rows.parquet')
    qrows, groups, context = context_from_cache(cache,rows)
    names = list(SPECS) if validation else [n for n,w in report['selection']['member_weights'].items() if w > 0 and n != 'reference']
    for name in names:
        if not validation:
            verify_complete(out/'validation'/name,signature)
        folder = out/stage/name
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():
            verify_complete(folder,signature)
            print(f'Reusing completed {name}',flush=True)
            continue
        X = append_features(pd.read_parquet(cache/'X.parquet'),rows,pd.read_parquet(features/'event_composition.parquet'))
        X = feature_view(X,SPECS[name]['view'])
        print(f'Fitting {name}: {X.shape[1]} features',flush=True)
        prior = None if validation else read(out/'validation'/name/'model_info.json')['rounds']
        p,info = fit_predict(X,rows,read(cache/'schema.json'),SPECS[name],folder,prior)
        info['columns'] = list(X)
        del X
        gc.collect()
        write(folder/'model_info.json',info)
        save_probability_artifact(folder/'probabilities.npz',qrows.txn_id,p,
            provenance={'signature':signature,'stage':stage,'member':name,'fit_ids_hash':ids_hash(rows.loc[rows['fit'],'txn_id'])})
        write(folder/'complete.json',{'signature':signature,'hashes':{f.name:file_hash(f) for f in folder.iterdir() if f.is_file() and f.name != 'complete.json'}})
    from compare_saved_models import load_stage, mix
    from postprocess_saved import align_ids
    _, old_meta, old_probs, old_groups, old_ctx = load_stage(root/'runs/local_podium/ensemble'/stage,stage)
    idx = align_ids(old_meta.txn_id,qrows.txn_id)
    probabilities = {'reference':mix(old_probs,[1/3]*3)[idx]}
    # Use the same reference rule context for all candidates, aligned by ID.
    groups, context = old_groups[idx], slice_rule_context(old_ctx,idx)
    del old_probs
    for name in names:
        probabilities[name] = load_probability_artifact(out/stage/name/'probabilities.npz',qrows.txn_id,
            expected_provenance={'signature':signature,'stage':stage,'member':name,'fit_ids_hash':ids_hash(rows.loc[rows['fit'],'txn_id'])})
    folder = out/stage
    if validation:
        dates = pd.DatetimeIndex(pd.to_datetime(qrows.timestamp,utc=True))
        cal = (dates.month == 9)&(dates.day>=16)&(dates.day<=23)
        assess = (dates.month == 9)&(dates.day>=24)&(dates.day<=30)
        # Exclude unconverged linear fits from the policy search.
        probabilities = {k:p for k,p in probabilities.items() if k=='reference' or read(folder/k/'model_info.json').get('converged',True)}
        from compare_diverse_models import GATE_CONFIG
        offsets,selection = select_member_policy({k:p[cal] for k,p in probabilities.items()},
            {k:float(k=='reference') for k in probabilities},qrows.y.to_numpy(dtype=int)[cal],groups[cal],slice_rule_context(context,cal),
            config={'min_gain':.002,'calibration':GATE_CONFIG,'baseline_calibration':GATE_CONFIG})
        write(folder/'selection.json',selection)
        weights = selection['member_weights']
        p = sum(weights[k]*v for k,v in probabilities.items())
        pred = predict_labels(p,groups,offsets,context)
        metrics = evaluate_predictions(qrows.y.to_numpy(dtype=int)[assess],pred[assess])
        baseline = evaluate_predictions(qrows.y.to_numpy(dtype=int)[assess],predict_labels(probabilities['reference'][assess],groups[assess],read(reference/'offsets.json'),slice_rule_context(context,assess)))
        if abs(baseline['macro_f1']-base_manifest['reference_f1']) > 1e-12:
            raise ValueError('saved baseline did not reproduce')
        report = {'selection':selection,'assessment':metrics,'baseline':baseline,'public_score':None,
            'assessment_reused_project_data':True,'eligible_for_refit':metrics['macro_f1']>baseline['macro_f1']+.002 and weights.get('reference',0)<1}
        write(folder/'report.json',report)
        write(folder/'offsets.json',offsets)
        lines = [f"Assessment macro-F1: {metrics['macro_f1']:.6f}",f"Reference: {baseline['macro_f1']:.6f}",
            f"Selected weights: {weights}",f"Eligible for refit: {report['eligible_for_refit']}",
            'Original labels; reused assessment data. Public score unknown.','Class: precision / recall / F1 / support']
        lines += [f"{k}: {v['precision']:.4f} / {v['recall']:.4f} / {v['f1']:.4f} / {v['support']}" for k,v in metrics['per_class'].items()]
        (out/'RUN_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('\n'.join(lines),flush=True)
    else:
        weights = report['selection']['member_weights']
        p = sum(weights.get(k,0)*v for k,v in probabilities.items())
        pred = predict_labels(p,groups,read(out/'validation/offsets.json'),context)
        result = write_submission(qrows.txn_id.to_numpy(dtype=str),pred,pd.read_csv(root/'data/raw/sample_submission.csv'),folder/'submission.csv')
        restored = pd.read_csv(folder/'submission.csv',dtype={'id':str})
        expected = pd.Series(np.asarray(LABELS)[pred],index=qrows.txn_id.astype(str)).loc[restored.id].to_numpy()
        if not np.array_equal(restored.label,expected):
            raise ValueError('submission read-back mismatch')
        write(folder/'submission_report.json',result)
        print(f'Saved {folder / "submission.csv"}',flush=True)
    write(folder/'complete.json',{'signature':signature,'hashes':{f.name:file_hash(f) for f in folder.iterdir() if f.is_file() and f.name!='complete.json'}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
