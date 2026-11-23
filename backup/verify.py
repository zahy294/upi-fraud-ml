import hashlib,json,os,subprocess,tarfile
from pathlib import Path
root=Path('/home/thanmay/ML-challenge');base=Path(__file__).resolve().parent
chunks=sorted((base/'chunks').glob('workspace.tar.zst.part-*'))
process=subprocess.Popen(['bash','-o','pipefail','-c','cat backup/chunks/workspace.tar.zst.part-* | zstd -d'],cwd=base.parent,stdout=subprocess.PIPE)
records=[]
with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
 for member in archive:
  original=root/member.name
  if member.isfile():
   h=hashlib.sha256();f=archive.extractfile(member)
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
   live=hashlib.sha256()
   with original.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''):live.update(b)
   assert h.digest()==live.digest(),member.name
   records.append({'path':member.name,'bytes':member.size,'sha256':h.hexdigest()})
  elif member.issym():
   assert os.readlink(original)==member.linkname,member.name
   records.append({'path':member.name,'symlink':member.linkname})
  else:raise AssertionError(member.name)
assert process.wait()==0
assert {r['path'] for r in records}==set((base/'files.list').read_text().splitlines())
(base/'file-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
(base/'verification.json').write_text(json.dumps({'passed':True,'files_verified':len(records),'chunks':len(chunks),'archive_bytes':sum(p.stat().st_size for p in chunks),'method':'Stream-decompress complete tar; compare every archived file SHA-256 and symlink target against live originals.'},indent=2)+'\n')
with (base/'SHA256SUMS').open('w') as f:
 for p in chunks+[base/'directory-symlinks.tar',base/'git-history.bundle',base/'file-manifest.json']:
  h=hashlib.sha256()
  with p.open('rb') as src:
   for block in iter(lambda:src.read(1048576),b''):h.update(block)
  f.write(h.hexdigest()+'  '+str(p.relative_to(base.parent))+'\n')
print((base/'verification.json').read_text())
