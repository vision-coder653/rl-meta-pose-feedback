"""Internal command feedback must replay PPO likelihoods without robot sensors."""
from pathlib import Path
import sys
APP = Path(__file__).resolve().parents[3]
for path in (APP, APP / "core", APP / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import unittest
from unittest.mock import patch
import numpy as np
import torch

from rl_meta_pose_feedback.tests import test_frozen_tracking as fixtures
from rl_meta_pose_feedback.models.action_distribution import MaskedMotorNormal


class PoseFeedbackTests(unittest.TestCase):
    def test_command_history_matches_physics_with_noise_and_clipping(self):
        config, actor, critic, norm, rollout, episodes, trainer = fixtures.FrozenTrackingTests().fixture()
        actions = torch.randn(2, config.episode_steps, 20) * .3
        actions[:, 0] = 10
        actions[:, 2] = -10
        history = rollout.command_history(episodes, actions).numpy()
        for b, episode in enumerate(episodes):
            state = rollout.environment.reset(episode.initial_angles)
            for t in range(config.episode_steps):
                np.testing.assert_array_equal(history[b, t], state.target_angles)
                state = rollout.environment.step(actions[b, t].numpy(), physics_steps=config.physics_steps).state
        changed = actions.clone()
        changed[:, -1] += 100
        # A current/future action must not leak into its own policy observation.
        np.testing.assert_array_equal(rollout.command_history(episodes, changed), history)

    def test_all_sampled_probabilities_replay_from_recorded_action_history(self):
        _, actor, _, norm, rollout, episodes, trainer = fixtures.FrozenTrackingTests().fixture()
        encoded = rollout.encode(actor, norm, episodes)
        actions, old_logp = rollout.actions(actor, norm, episodes, stochastic=True, encoded=encoded)
        history = rollout.command_history(episodes, actions).float()
        mask = torch.as_tensor(rollout.mask)
        with torch.no_grad():
            command = actor.decode_latent(encoded.latent, torch.as_tensor(rollout.morphology), mask,
                command_pose=history)
            replay = MaskedMotorNormal(command.motor_increment_mean, mask,
                                        std=actor.motor_noise.stddev).log_prob(actions)
        torch.testing.assert_close(replay, old_logp, atol=1e-4, rtol=0)
        mean_actions, _ = rollout.actions(actor, norm, episodes, stochastic=False, encoded=encoded)
        self.assertGreater((history-rollout.command_history(episodes, mean_actions)).abs().max().item(), .001)

    def test_sampling_never_reads_physics_and_resets_command_history(self):
        _, actor, _, norm, rollout, episodes, _ = fixtures.FrozenTrackingTests().fixture()
        with patch.object(rollout.environment, "reset", side_effect=AssertionError("No physics in actor")), \
             patch.object(rollout.environment, "step", side_effect=AssertionError("No physics in actor")), \
             patch.object(rollout.environment, "state", side_effect=AssertionError("No measured state")):
            before, _ = rollout.actions(actor, norm, episodes, stochastic=False)
            rollout.actions(actor, norm, episodes[::-1], stochastic=True)
            after, _ = rollout.actions(actor, norm, episodes, stochastic=False)
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_fusion_uses_command_pose_and_accepts_single_or_sequence_steps(self):
        _, actor, _, norm, rollout, episodes, _ = fixtures.FrozenTrackingTests().fixture()
        encoded = rollout.encode(actor, norm, episodes)
        morphology = torch.as_tensor(rollout.morphology)
        command_pose = rollout.command_history(episodes, torch.zeros(2, 8, 20)).float()
        head = actor.command_head
        args = dict(command_pose=command_pose, motor_mask=torch.as_tensor(rollout.mask))
        base = head(encoded.latent, morphology, **args)
        changed = head(encoded.latent, morphology, **{**args, "command_pose": command_pose + .2})
        self.assertGreater((changed-base).abs().max().item(), 1e-7)
        self.assertFalse(hasattr(head, "human_projection"))
        single = head(encoded.latent[:, 0], morphology, motor_mask=args["motor_mask"],
                      command_pose=command_pose[:, 0])
        torch.testing.assert_close(single, base[:, 0], atol=1e-7, rtol=1e-5)
        base.sum().backward()
        for branch in (head.latent_projection, head.command_projection):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()))

    def test_missing_joints_are_zero_masked_without_shifting_other_slots(self):
        from simulation.morphology import hod_policy_features
        _, actor, _, _, rollout, episodes, _ = fixtures.FrozenTrackingTests().fixture()
        # Partial fingers and an entirely missing finger in canonical HOD slots.
        morphology = torch.from_numpy(hod_policy_features(("990", "90", "0", "", "990")))
        mask = torch.cat((morphology[:5, None], morphology[5:].reshape(5, 6)[:, :3]), -1).bool().flatten()
        actor.training_group = None  # This unit test isolates variable morphology.
        latent = torch.randn(2, 8, 512)
        pose = torch.zeros(2, 8, 20, requires_grad=True)
        before = actor.decode_latent(latent, morphology, mask, command_pose=pose)
        dirty = pose.detach().clone()
        dirty[..., ~mask] = float("nan")
        after = actor.decode_latent(latent, morphology, mask, command_pose=dirty)
        torch.testing.assert_close(before.motor_increment_mean, after.motor_increment_mean, atol=0, rtol=0)
        self.assertTrue((before.motor_increment_mean[..., ~mask] == 0).all())
        before.coefficients.sum().backward()
        self.assertTrue((pose.grad[..., ~mask] == 0).all())
        with self.assertRaisesRegex(ValueError, "joint presence"):
            actor.decode_latent(latent, morphology, torch.ones(20, dtype=torch.bool), command_pose=pose)
        # Presence bits, rather than angle value, distinguish absent from zero.
        full = torch.from_numpy(hod_policy_features(("990",) * 5))
        real_zero = actor.decode_latent(latent, full, torch.ones(20, dtype=torch.bool), command_pose=pose)
        self.assertGreater((real_zero.coefficients-before.coefficients).abs().max().item(), 1e-7)
        rollout.mask = mask.numpy()
        actions = torch.full((2, 8, 20), 10.)
        history = rollout.command_history(episodes, actions)
        self.assertTrue((history[..., ~mask] == 0).all())


if __name__ == "__main__":
    unittest.main()
