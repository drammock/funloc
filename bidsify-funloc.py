"""Create BIDS folder structure for "funloc" data."""

import tarfile
from pathlib import Path
from warnings import filterwarnings

import numpy as np
import mne
from mne_bids import (
    BIDSPath,
    get_anat_landmarks,
    make_dataset_description,
    write_anat,
    write_meg_calibration,
    write_meg_crosstalk,
    write_raw_bids,
)
from mnefun import extract_expyfun_events

mne.set_log_level("WARNING")
# suppress messages about IAS / MaxShield
filterwarnings(
    action="ignore",
    message="This file contains raw Internal Active Shielding data",
    category=RuntimeWarning,
    module="mne",
)

# path stuff
root = Path("/data/funloc").resolve()
meg_dir = root / "meg"
cal_dir = root / "calibration"
bids_root = root / "bids-data"
derivs_subj_dir = bids_root / "derivatives" / "freesurfer" / "subjects"
derivs_subj_dir.mkdir(parents=True, exist_ok=True)

bids_path = BIDSPath(root=bids_root, datatype="meg", suffix="meg", extension=".fif")
read_raw_kw = dict(allow_maxshield=True, preload=False, verbose=False)

for subj_num in (1, 2):
    subj = f"subj_{subj_num:02}"
    raw_file = meg_dir / subj / "raw_fif" / f"{subj}_funloc_raw.fif"
    erm_file = meg_dir / subj / "raw_fif" / f"{subj}_erm_raw.fif"
    raw = mne.io.read_raw_fif(raw_file, **read_raw_kw)
    erm = mne.io.read_raw_fif(erm_file, **read_raw_kw)
    # bad channels
    mf_prebad = ["MEG0743", "MEG1442"]
    if subj_num == 1:
        mf_prebad.extend(["MEG0422", "MEG0743", "MEG2011", "MEG2532"])
    raw.info["bads"] = mf_prebad
    bads_file = meg_dir / subj / "bads" / f"bad_ch_{subj}_post-sss.txt"
    if bads := bads_file.read_text():
        raw.info["bads"].extend(bads.split("\n"))
    # events
    events, *_ = extract_expyfun_events(raw_file)
    event_id = {
        "auditory/standard": 10,
        "auditory/deviant": 14,
        "visual/standard": 12,
        "visual/deviant": 16,
    }
    # adjust for known delay in trigger timing
    t_adjust = -4.0e-3  # triggers known to have 4ms delay
    s_adjust = np.rint(t_adjust * raw.info["sfreq"]).astype(int)
    events[:, 0] += s_adjust
    assert np.all(events[:, 0] > 0)

    # write MEG
    bids_path.update(subject=f"{subj_num:02}", task="funloc")
    write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        events=events,
        event_id=event_id,
        empty_room=erm,
        overwrite=True,
    )
    # extract MRI to derivs tree
    fs_subject = f"sub{subj_num:02}"
    mri_name = f"AKCLEE_{107 if subj_num == 1 else 110}_slim"
    mri_archive = root / "mri" / f"{mri_name}.tar.gz"
    tar = tarfile.open(mri_archive)
    tar.extractall(path=derivs_subj_dir / fs_subject, filter="data")
    tar.close()
    # remove containing folder
    container = derivs_subj_dir / fs_subject / mri_name
    for item in container.iterdir():
        item.replace(derivs_subj_dir / fs_subject / item.relative_to(container))
    container.rmdir()
    # rename "AKCLEE_NNN_slim" → "subNN"
    for dirpath, dirnames, filenames in (derivs_subj_dir / fs_subject).walk():
        for fname in filenames:
            if mri_name in fname:
                new_fname = fname.replace(mri_name, fs_subject)
                (dirpath / fname).replace(dirpath / new_fname)
    # write anat
    t1_fname = derivs_subj_dir / fs_subject / "mri" / "T1.mgz"
    trans = mne.read_trans(meg_dir / subj / "trans" / f"{subj}-trans.fif")
    landmarks = get_anat_landmarks(
        image=t1_fname,
        info=raw.info,
        trans=trans,
        fs_subject=fs_subject,
        fs_subjects_dir=derivs_subj_dir,
    )
    anat_bids_path = BIDSPath(root=bids_root, subject=f"{subj_num:02}")
    nii_file = write_anat(image=t1_fname, bids_path=anat_bids_path, landmarks=landmarks)
    # write the fine-cal and crosstalk files (once per subject)
    write_meg_calibration(cal_dir / "sss_cal.dat", bids_path=anat_bids_path)
    write_meg_crosstalk(cal_dir / "ct_sparse.fif", bids_path=anat_bids_path)


make_dataset_description(
    path=bids_root,
    name="MNE funloc data",
    authors=["Eric Larson", "Ross Maddox", "Daniel McCloy"],
    overwrite=True,
)

with tarfile.open(root / "MNE-funloc-data.tar.gz", "w:gz") as tar:
    for name in bids_root.iterdir():
        tar.add(name, arcname=Path("MNE-funloc-data") / name.relative_to(bids_root))
