"""Download a pinned official archive and extract only the verified tracking model."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz'
ARCHIVE_SHA256 = '164e0bcbb52c25b116e77b799d8eb81c59994090608c73de3946c87e34c3db66'
MODEL_SHA256 = '629783b2b1e8bc1316190fc2a57ba95396dd0d6b59cfbb38543fc6ad8bd61ae5'
MODEL_NAME = 'tracking_vemg2pose.ckpt'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Official archive checksum mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if sha256(output) == MODEL_SHA256:
            return
        raise ValueError('Refusing to overwrite an existing file with a different checksum')
    handle = tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + '.', suffix='.tmp', delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        with tarfile.open(archive, 'r:gz') as source:
            members = [m for m in source.getmembers() if m.isfile()
                       and PurePosixPath(m.name).name == MODEL_NAME]
            if len(members) != 1:
                raise ValueError('Expected exactly one official tracking checkpoint')
            # No extractall: archive paths never select filesystem destinations.
            with source.extractfile(members[0]) as payload, temporary.open('wb') as target:
                shutil.copyfileobj(payload, target)
        if sha256(temporary) != MODEL_SHA256:
            raise ValueError('Extracted tracking model checksum mismatch')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official archive')
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/meta' / MODEL_NAME)
    args = parser.parse_args()
    if args.output.exists():
        if sha256(args.output) != MODEL_SHA256:
            parser.error('Existing output has an unexpected checksum; it was not overwritten')
        print(f'Already verified: {args.output}')
        return
    archive = args.archive
    if archive is None:
        archive = ROOT / '.cache/emg2pose_model_checkpoints.tar.gz'
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            temporary = archive.with_suffix('.download.tmp')
            try:
                print('Downloading official checkpoint archive (about 248 MB)...', flush=True)
                with urlopen(URL, timeout=60) as response, temporary.open('wb') as target:
                    shutil.copyfileobj(response, target)
                if sha256(temporary) != ARCHIVE_SHA256:
                    raise ValueError('Download checksum mismatch')
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
    prepare(archive, args.output)
    print(f'Verified tracking checkpoint: {args.output}')


if __name__ == '__main__':
    main()
