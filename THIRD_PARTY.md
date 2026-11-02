# Third-party source and model resources

- **Meta emg2pose / UmeTrack** — source repository: https://github.com/facebookresearch/emg2pose, revision `5f6f62b1a0a08426adffe55900842e75a8adb38c`.
  - `third_party/meta_reference/emg2pose/{__init__,constants,networks,pose_modules}.py` are official reference files used only for numerical-equivalence tests. The upstream license is `third_party/meta_reference/LICENSE` (CC BY-NC-SA 4.0).
  - `core/kinematics/umetrack/` contains the pinned FK implementation; `core/resources/umetrack/` contains generic hand geometry, `LICENSE`, and detailed `UPSTREAM.md` provenance.
  - `experiments/rl_meta_pose_feedback/models/emg2pose_tds.py` and the tracking adapter derive from the same upstream architecture. See `core/resources/licenses/emg2pose_LICENSE.txt`; preserve source notices.
  - Pretrained weights are downloaded separately from the official host. The repository includes no checkpoint or EMG recording.
- **House of Dextra** — source repository: https://github.com/An-Axolotl/House-Of-Dextra-XEmbodied-CoDesign, revision `20c1ed7ff50fc36fe170066ec185a1310ce7fa0f`.
  - `resources/robot/` contains the fixed generated hand and only its required XML/mesh dependencies. Generation settings are in `resources/robot/generation_config.json`.
  - The original MIT license is retained as `resources/robot/LICENSE`.

This file records component provenance; it does not replace or relicense the upstream license texts.
