"""Run directly with the experiment Python environment; no training dataset needed."""
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[3]
for source in (APP, APP / "core", APP / "experiments"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from dataclasses import replace
import importlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from emg_control_paths import CACHE_DIR, EXPERIMENT_DIR, bundle_path
from rl_meta_pose_feedback.models.encoder import EMGEncoder, EncoderConfig
from rl_meta_pose_feedback.training.config import TrainingConfig, PPOConfig
from rl_meta_pose_feedback.training.runtime import create_runtime
from rl_meta_pose_feedback.training.trainer import PPOTrainer
from rl_meta_pose_feedback.training.run import save_checkpoint


class FrozenTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.encoder = EMGEncoder().eval()

    def sample(self, steps, offset=0.):
        count = 1790 + 1 + 40 * (steps + 1)
        time = 10 + torch.arange(count, dtype=torch.float64) / 2000
        angles = (.1 * torch.sin(torch.arange(count).float()[:, None] / 200) + offset).expand(-1, 20).clone()
        return {"emg": torch.randn(count, 16), "time": time, "joint_angles": angles,
                "max_gap_seconds": .00075}

    def fixture(self):
        config = TrainingConfig(device="cpu", episode_steps=8, episodes_per_update=2,
                                total_updates=1, episode_initialization="fixed",
                                ppo=PPOConfig(update_epochs=2, minibatch_episodes=1, target_kl=1.))
        actor, critic, normalizer, rollout = create_runtime(config)
        episodes = [rollout.prepare(self.sample(config.episode_steps, offset), left_context=1790, stride=40)
                    for offset in (0., .1)]
        trainer = PPOTrainer(actor, critic, normalizer, rollout, config)
        return config, actor, critic, normalizer, rollout, episodes, trainer

    def test_official_tracking_pose_and_hidden_equivalence(self):
        source = EXPERIMENT_DIR / "third_party/meta_reference"
        sys.path.insert(0, str(source))
        networks = importlib.import_module("emg2pose.networks")
        poses = importlib.import_module("emg2pose.pose_modules")
        # Independent official CNN/TDS and SequentialLSTM implementations.
        net = networks.TdsNetwork(
            conv_blocks=[networks.Conv1dBlock(16, 256, kernel_size=11, stride=5),
                         networks.Conv1dBlock(256, 256, kernel_size=5, stride=2)],
            tds_stages=[
                networks.TdsStage(in_channels=256, in_conv_kernel_width=17, in_conv_stride=4,
                    num_blocks=2, channels=16, feature_width=16, kernel_width=9),
                networks.TdsStage(in_channels=256, in_conv_kernel_width=9, in_conv_stride=2,
                    num_blocks=2, channels=16, feature_width=16, kernel_width=5, out_channels=64)])
        decoder = networks.SequentialLSTM(84, 20, 512, num_layers=2, scale=.01)
        official = poses.StatePoseModule(net, decoder, state_condition=True, predict_vel=True, rollout_freq=50).eval()
        official.load_state_dict(self.encoder.state_dict(), strict=True)
        emg = self.sample(256)["emg"][None, :1790+40*256]
        q0 = torch.linspace(-.1, .1, 20)[None]
        recorded = []
        hook = decoder.lstm.register_forward_hook(lambda module, args, result: recorded.append(result[0][:, 0]))
        with torch.no_grad():
            expected = official._predict_pose(emg.transpose(1, 2), q0).transpose(1, 2)
        hook.remove()
        actual = self.encoder(emg, q0, steps=256)
        torch.testing.assert_close(actual.predicted_pose, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual.latent, torch.stack(recorded, 1), atol=1e-6, rtol=1e-6)
        self.assertEqual(actual.latent.shape, (1, 256, 512))
        # Explicitly disclose that official linear interpolation can look ahead.
        self.assertTrue((actual.feature_available_indices > actual.sample_indices).any())

    def test_guard_tail_reset_and_initial_pose(self):
        emg = self.sample(12)["emg"][None]
        q0 = torch.zeros(1, 20)
        expected = self.encoder(emg, q0, steps=12)
        altered = emg.clone()
        altered[:, 1790+40*12:] = 10000
        self.encoder(emg, q0 + 1, steps=12)
        replay = self.encoder(altered, q0, steps=12)
        torch.testing.assert_close(replay.latent, expected.latent, rtol=0, atol=0)
        changed = self.encoder(emg, q0 + .4, steps=12)
        self.assertGreater((changed.latent - expected.latent).abs().max().item(), 1e-5)
        self.encoder.train()
        self.assertFalse(self.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in self.encoder.parameters()))

    def test_rollout_uses_only_initial_label_and_50hz_physics(self):
        config, actor, _, norm, rollout, episodes, _ = self.fixture()
        before, _ = rollout.actions(actor, norm, episodes, stochastic=False)
        for episode in episodes:
            torch.testing.assert_close(episode.human_initial_pose, episode.reference_angles[0])
            episode.reference_angles[1:] += 2
            episode.sample["joint_angles"] += 3
        after, _ = rollout.actions(actor, norm, episodes, stochastic=False)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        trace = rollout.execute(episodes[0], torch.zeros(config.episode_steps, 20)).trace
        np.testing.assert_allclose(trace["simulation_time"], np.arange(1, config.episode_steps+1)*.02, atol=1e-10)
        np.testing.assert_allclose(np.diff(episodes[0].query_times), .02, atol=1e-10)

    def test_ppo_freezes_meta_reuses_features_and_updates_controller(self):
        _, actor, critic, norm, rollout, episodes, trainer = self.fixture()
        components = {"encoder": actor.encoder, "command_head": actor.command_head,
                      "synergy": actor.synergy, "motor_noise": actor.motor_noise, "critic": critic}
        before = {key: [p.detach().clone() for p in mod.parameters()] for key, mod in components.items()}
        optimized = {id(p) for g in trainer.actor_optimizer.param_groups for p in g["params"]}
        self.assertFalse(optimized.intersection(id(p) for p in actor.encoder.parameters()))
        with patch.object(actor.encoder, "forward", wraps=actor.encoder.forward) as forward:
            actions, logp, outcomes = trainer.collect(episodes)
            replay_logp = rollout.actions(actor, norm, episodes, stochastic=False)[1]
            # The policy check during update verifies probabilities of the sampled actions.
            self.assertTrue(torch.isfinite(replay_logp).all())
            calls = forward.call_count
            metrics = trainer.update(episodes, actions, logp, outcomes)
            self.assertEqual(forward.call_count, calls)
        self.assertGreater(metrics["optimizer_steps"], 0)
        self.assertEqual(metrics["gradient_norms"]["encoder"], 0.)
        for key, mod in components.items():
            unchanged = all(torch.equal(p, old) for p, old in zip(mod.parameters(), before[key]))
            self.assertEqual(unchanged, key == "encoder", key)
        # Controls must recompute their own features, keeping q0 fixed.
        zero = [replace(e, sample={**e.sample, "emg": torch.zeros_like(e.sample["emg"])}) for e in episodes]
        normal = rollout.encode(actor, norm, episodes).latent
        silent = rollout.encode(actor, norm, zero).latent
        self.assertGreater((normal - silent).abs().max().item(), 1e-6)

    def test_checkpoint_load_reproduces_evaluation_without_official_reload(self):
        config, actor, critic, norm, rollout, episodes, trainer = self.fixture()
        trainer.update(episodes, *trainer.collect(episodes))
        folder = CACHE_DIR / "rl_meta_pose_feedback_tests"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            path = Path(temporary) / "last.pt"
            save_checkpoint(path, config, actor, critic, norm, rollout, trainer, update=1, pools={"train": [], "val": []})
            state = torch.load(path, map_location="cpu", weights_only=True)
        with patch.object(EMGEncoder, "load_official_weights", side_effect=AssertionError("Checkpoint must be self-contained")):
            actor2, critic2, norm2, rollout2 = create_runtime(config, state)
        with torch.no_grad():
            first = rollout.actions(actor, norm, episodes, stochastic=False)[0]
            second = rollout2.actions(actor2, norm2, episodes, stochastic=False)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for p, q in zip(critic.parameters(), critic2.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "different experiment"):
            create_runtime(config, {**state, "experiment_id": "rl_decoder"})

    def test_rejects_wrong_frequency_preprocessing_or_unfreeze(self):
        for values in ({"physics_steps": 20}, {"nominal_sample_rate": 1000}, {"emg_preprocessing": "standardize"}):
            with self.assertRaises(ValueError):
                TrainingConfig(**values)
        with self.assertRaises(ValueError):
            EncoderConfig(frozen=False)

    def test_tracking_reward_removes_open_prior_without_changing_physics(self):
        config, actor, critic, norm, rollout, episodes, trainer = self.fixture()
        actions = torch.full((config.episode_steps, 20), .01)
        current = rollout.execute(episodes[0], actions)
        np.testing.assert_array_equal(current.trace["costs"], current.trace["geometric_costs"])
        self.assertEqual(float(current.trace["regularization_costs"].sum()), 0.)
        self.assertGreater(np.abs(current.trace["actual_joint_angles"]).max(), .01)
        self.assertEqual(config.retargeting.open_regularization, .0025)
        historical = replace(config, reward_objective="retarget")
        _, _, _, old_rollout = create_runtime(historical)
        previous = old_rollout.execute(episodes[0], actions)
        np.testing.assert_array_equal(current.trace["actual_joint_angles"], previous.trace["actual_joint_angles"])
        np.testing.assert_array_equal(current.trace["geometric_costs"], previous.trace["geometric_costs"])
        self.assertGreater(float(previous.trace["regularization_costs"].sum()), 0.)
        np.testing.assert_allclose(previous.trace["costs"],
                                   current.trace["costs"] + previous.trace["regularization_costs"], rtol=0, atol=0)
        # Training, evaluation and replay refuse to relabel historical weights.
        folder = CACHE_DIR / "rl_meta_pose_feedback_tests"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            path = Path(temporary) / "old.pt"
            save_checkpoint(path, historical, actor, critic, norm, old_rollout, trainer,
                            update=0, pools={"train": [], "val": []})
            state = torch.load(path, map_location="cpu", weights_only=True)
        restored = TrainingConfig.from_dict(state["config"])
        _, _, _, restored_rollout = create_runtime(restored, state)
        self.assertEqual(restored_rollout.reward_settings["open_regularization"], .0025)
        with self.assertRaisesRegex(ValueError, "reward objective differs"):
            create_runtime(config, state)

    def test_current_configs_use_geometry_only_reward(self):
        for filename in ("ppo.json", "ppo_raw_large.json"):
            config = TrainingConfig.load(bundle_path("rl_meta_pose_feedback", "configs", filename))
            self.assertEqual(config.reward_objective, "tracking")
            self.assertEqual(config.ppo.kl_control, "adaptive_lr")
            self.assertEqual(config.retargeting.open_regularization, .0025)


if __name__ == "__main__":
    unittest.main()
