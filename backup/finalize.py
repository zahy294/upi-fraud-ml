import subprocess,json
from pathlib import Path
base=Path(__file__).resolve().parent;repo=base.parent
assert json.loads((base/'verification.json').read_text())['passed']
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0',*args],cwd=repo,text=True).strip()
commits=[git('rev-parse','HEAD')]
chunks=sorted((base/'chunks').glob('workspace.tar.zst.part-*'))
for offset in range(0,len(chunks),5):
 batch=chunks[offset:offset+5]
 git('add','-f',*[str(p.relative_to(repo)) for p in batch])
 git('commit','-m',f'Back up verified workspace archive chunks {offset+1}-{offset+len(batch)} of {len(chunks)}')
 commits.append(git('rev-parse','HEAD'))
script='#!/usr/bin/env bash\nset -euo pipefail\ncd "'+str(repo)+'"\nbranch="refs/heads/thanmay/full-backup-2026-11-24"\n'
script+='remote_head=$(git ls-remote origin "$branch" | cut -f1)\n'
for h in commits:
 script+='if [ -n "$remote_head" ] && git merge-base --is-ancestor '+h+' "$remote_head"; then echo "Already backed up '+h+'"; else git push origin '+h+':"$branch"; remote_head='+h+'; fi\n'
script+='git push -u origin thanmay/full-backup-2026-11-24\nlocal_head=$(git rev-parse HEAD)\nremote_head=$(git ls-remote origin "$branch" | cut -f1)\ntest "$local_head" = "$remote_head"\necho "Remote branch verified at $remote_head"\n'
(base/'push-backup.sh').write_text(script);(base/'push-backup.sh').chmod(0o755)
git('add','-f',*[str(p.relative_to(repo)) for p in base.iterdir() if p.is_file() and p.name not in {'finalize.py'}])
git('add','-f','backup/finalize.py')
git('commit','-m','Add complete backup integrity manifest and batched authenticated push instructions')
print('FINAL HEAD:',git('rev-parse','HEAD'));print('CHUNKS:',len(chunks));print('STATUS:',git('status','--short'))
