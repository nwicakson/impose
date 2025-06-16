#!/usr/bin/env python3
"""
Upload checkpoints and logs from impose/exp to Google Drive.

Local tree (example)
└─ impose/exp/
   ├─ folder1/ ckpt_best.pth | config.yml | stdout.txt
   ├─ folder2/ ckpt_best.pth | config.yml | stdout.txt
   ├─ *.out
   └─ *.out
Remote tree created in Drive
└─ BACKUP_PARENT/
   ├─ folder1/ ckpt_best.pth | config.yml | stdout.txt
   ├─ folder2/ ckpt_best.pth | config.yml | stdout.txt
   ├─ *.out
   └─ *.out
"""

from pathlib import Path
from tqdm import tqdm
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import argparse, hashlib, sys

# -------------  EDIT THESE TO CHANGE FOLDER -----------------------------------
LOCAL_ROOT = Path("/mnt/extended-home/nugroho/program/code/impose/exp")
DRIVE_PARENT = "1R1JFX8Wx2e190aIHvVhCUlqLM18l2nLo"   # paste folder-ID from Drive
#LOCAL_ROOT = Path("/mnt/extended-home/nugroho/program/code/diffpose-nw_ori/exp")
#DRIVE_PARENT = "13q-9gTNJB2gLKWjjfRBX23R1-O9N0X8l"
# ------------------------------------------------------------------------------

# ---------- auth paths -------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent          # /…/code
SECRET_FILE = BASE_DIR / "client_secrets.json"               # json file you downloaded
TOKEN_FILE  = BASE_DIR / "credentials.json"                  # gets created after 1st auth

# ---------- authenticate ----------------------------------------------
gauth = GoogleAuth()                                         # ← no arg here
gauth.LoadClientConfigFile(str(SECRET_FILE))                 # point at the JSON
gauth.LoadCredentialsFile(str(TOKEN_FILE))                   # may not exist yet

if not gauth.credentials:
    gauth.CommandLineAuth()          # prints URL → paste code once
    gauth.SaveCredentialsFile(str(TOKEN_FILE))

drive = GoogleDrive(gauth)

# ----  Helper: find or create a sub-folder on Drive  --------------------------
def ensure_drive_folder(drive, name: str, parent_id: str) -> str:
    """Return the folder-ID for *name* inside *parent_id*, creating it if needed."""
    query = (
        f"'{parent_id}' in parents and trashed=false "
        f"and mimeType='application/vnd.google-apps.folder' and title='{name}'"
    )
    hits = drive.ListFile({"q": query, "maxResults": 1}).GetList()
    if hits:
        return hits[0]["id"]

    folder = drive.CreateFile(
        {
            "title": name,
            "parents": [{"id": parent_id}],
            "mimeType": "application/vnd.google-apps.folder",
        }
    )
    folder.Upload()
    return folder["id"]

# ----  Helper: upload a single file (overwrites if exists)  -------------------
# ---------------------------------------------------------------------------
# CLI: --on-conflict {skip,overwrite}
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Backup impose/exp to Google Drive."
)
parser.add_argument(
    "--skip",                       # ← nicer flag name now
    dest="on_conflict",
    choices=["skip", "overwrite"],
    default="overwrite",            # ← NEW DEFAULT
    help=("What to do when the file name already exists in Drive "
          "(default: overwrite if content differs; use --skip to leave it).")
)
ARGS = parser.parse_args()

def md5sum(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with path.open("rb") as fh:
        for piece in iter(lambda: fh.read(chunk), b""):
            h.update(piece)
    return h.hexdigest()


def upload_file(local_path: Path, parent_id: str):
    escaped = local_path.name.replace("'", "\\'")
    query = (
        f"'{parent_id}' in parents and trashed=false "
        f"and title='{escaped}'"
    )
    matches = drive.ListFile({"q": query, "maxResults": 1}).GetList()

    if matches:
        remote = matches[0]

        # identical → always skip
        if remote.get("md5Checksum") == md5sum(local_path):
            print(f"· identical, skip {local_path.relative_to(LOCAL_ROOT)}")
            return

        # conflict policy
        if ARGS.on_conflict == "skip":
            print(f"· changed, skip (conflict policy) {local_path.relative_to(LOCAL_ROOT)}")
            return

        # overwrite
        gfile = drive.CreateFile({"id": remote["id"]})
        action = "update"

    else:  # first upload
        gfile = drive.CreateFile(
            {"title": local_path.name, "parents": [{"id": parent_id}]}
        )
        action = "upload"

    gfile.SetContentFile(str(local_path))
    gfile.Upload()
    print(f"↑ {action} {local_path.relative_to(LOCAL_ROOT)}")

# ----  1) Upload ckpt_best / config / stdout inside each folder-X  ------------
for sub in sorted(p for p in LOCAL_ROOT.iterdir() if p.is_dir()):
    dest_id = ensure_drive_folder(drive, sub.name, DRIVE_PARENT)
    for fname in ("ckpt_best.pth", "config.yml", "stdout.txt"):
        f = sub / fname
        if f.exists():
            upload_file(f, dest_id)

for out in sorted(LOCAL_ROOT.glob("*.out")):
    upload_file(out, DRIVE_PARENT)

print("✅  Backup finished.")
