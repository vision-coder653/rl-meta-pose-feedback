"""Sequence-preserving PPO-Clip with GAE and a separate privileged critic."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from rl_meta_pose_feedback.models.action_distribution import MaskedMotorNormal
from rl_meta_pose_feedback.training.config import TrainingConfig


class ValidationEarlyStopping:
    """Stop on a fixed validation pool; small cumulative gains can reset patience."""

    def __init__(self, initial_cost, config):
        self.reference = float(initial_cost)
        self.patience = config.early_stop_patience
        self.min_updates = config.early_stop_min_updates
        self.min_relative = config.early_stop_relative_improvement
        self.stale_checks = 0

    def observe(self, update, cost):
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Validation motion cost must be finite and nonnegative")
        if cost < self.reference - max(abs(self.reference), 1e-12)*self.min_relative:
            self.reference, self.stale_checks = float(cost), 0
        else:
            self.stale_checks += 1
        return bool(self.patience and update >= self.min_updates and self.stale_checks >= self.patience)

    def status(self):
        return {"meaningful_best_cost": self.reference, "stale_validation_checks": self.stale_checks,
                "patience": self.patience, "min_updates": self.min_updates,
                "minimum_relative_improvement": self.min_relative}


def generalized_advantage(rewards, values, terminated, truncated, *, gamma, gae_lambda):
    """[B,T] rewards/flags and [B,T+1] values, including final-state value.

    Terminations do not bootstrap. Truncations bootstrap their final state but
    stop the GAE recursion, so no return propagates through a reset boundary.
    """
    if (rewards.ndim != 2 or values.shape != (*rewards.shape[:1], rewards.shape[1]+1)
            or terminated.shape != rewards.shape or truncated.shape != rewards.shape
            or terminated.dtype != torch.bool or truncated.dtype != torch.bool
            or not torch.isfinite(rewards).all() or not torch.isfinite(values).all()):
        raise ValueError("Invalid GAE shapes, flags or values")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE gamma/lambda must be in [0,1]")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        delta = rewards[:, t] + gamma * (~terminated[:, t]) * values[:, t+1] - values[:, t]
        carry = delta + gamma * gae_lambda * (~(terminated[:, t] | truncated[:, t])) * carry
        advantages[:, t] = carry
    return advantages, advantages + values[:, :-1]


def clipped_policy_loss(new_log_prob, old_log_prob, advantages, *, clip_ratio):
    if new_log_prob.shape != old_log_prob.shape or advantages.shape != new_log_prob.shape:
        raise ValueError("PPO probability and advantage shapes must agree")
    log_ratio = new_log_prob - old_log_prob.detach()
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise RuntimeError("Nonfinite PPO probability ratio")
    loss = -torch.minimum(ratio * advantages.detach(),
                          ratio.clamp(1-clip_ratio, 1+clip_ratio) * advantages.detach()).mean()
    with torch.no_grad():
        kl = ((ratio-1) - log_ratio).mean()
        clip_fraction = ((ratio-1).abs() > clip_ratio).float().mean()
    return loss, kl, clip_fraction


def checked_grad_norm(parameters, *, name):
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters:
        return 0.0
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError(f"Missing or nonfinite {name} gradient")
    return float(torch.stack([p.grad.detach().square().sum() for p in parameters]).sum().sqrt())


def diagonal_gaussian_kl(old_mean, old_std, new_mean, new_std, mask):
    """Exact KL(old || new), summed over active motors then averaged over B,T."""
    terms = torch.log(new_std / old_std) + (
        old_std.square() + (old_mean - new_mean).square()
    ) / (2 * new_std.square()) - .5
    return (terms * mask.to(terms.dtype)).sum(-1).mean()


def adaptive_kl_learning_rate(current_lr, kl, ppo):
    """rl_games adaptive rule; the deadband avoids reacting to every fluctuation.

    See https://github.com/Denys88/rl_games/blob/master/rl_games/common/schedulers.py
    Only the actor is scheduled here because our critic has a separate optimizer.
    """
    if not math.isfinite(kl):
        raise RuntimeError("Nonfinite analytic policy KL")
    if kl > 2 * ppo.target_kl:
        current_lr /= ppo.kl_lr_factor
    elif kl < .5 * ppo.target_kl:
        current_lr *= ppo.kl_lr_factor
    return min(ppo.actor_max_learning_rate, max(ppo.actor_min_learning_rate, current_lr))


class PPOTrainer:
    def __init__(self, actor, critic, normalizer, rollout, config: TrainingConfig):
        self.actor, self.critic, self.normalizer = actor, critic, normalizer
        self.rollout, self.config = rollout, config
        ppo = config.ppo
        if any(p.requires_grad for p in actor.encoder.parameters()):
            raise ValueError("Frozen-feature PPO cannot train the Meta encoder; use sequence replay for fine-tuning")
        self._feature_cache = None
        self.actor_optimizer = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=ppo.actor_learning_rate, eps=ppo.adam_epsilon)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=ppo.critic_learning_rate, eps=ppo.adam_epsilon)
        self.rng = np.random.default_rng(config.seed + 101)
        # eval disables dropout; it does not disable parameter gradients.
        self.actor.eval()
        self.critic.train()

    def collect(self, episodes):
        self.actor.eval()
        encoded = self.rollout.encode(self.actor, self.normalizer, episodes)
        self._feature_cache = (tuple(episodes), encoded)
        raw_actions, old_log_prob = self.rollout.actions(self.actor, self.normalizer, episodes, stochastic=True, encoded=encoded)
        outcomes = [self.rollout.execute(episode, action) for episode, action in zip(episodes, raw_actions)]
        return raw_actions, old_log_prob, outcomes

    def update(self, episodes, raw_actions, old_log_prob, outcomes):
        config, ppo = self.config, self.config.ppo
        device = next(self.actor.parameters()).device
        observations = torch.stack([item.critic_observations for item in outcomes]).to(device)
        rewards = -config.reward_scale * torch.stack([item.costs for item in outcomes]).to(device)
        actions, old_log_prob = raw_actions.to(device), old_log_prob.to(device)
        with torch.no_grad():
            values = self.critic(observations)
            # A finite reference task ends here; its terminal continuation is 0.
            values_with_end = torch.cat((values, torch.zeros_like(values[:, :1])), dim=1)
            terminal = torch.zeros_like(rewards, dtype=torch.bool)
            terminal[:, -1] = True
            advantages, returns = generalized_advantage(rewards, values_with_end, terminal,
                torch.zeros_like(terminal), gamma=ppo.gamma, gae_lambda=ppo.gae_lambda)
            advantages = (advantages-advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        morphology = torch.as_tensor(self.rollout.morphology, device=device)
        motor_mask = torch.as_tensor(self.rollout.mask, device=device)
        cached = self._feature_cache
        same_batch = cached is not None and len(cached[0]) == len(episodes) and all(a is b for a, b in zip(cached[0], episodes))
        encoded = cached[1] if same_batch else self.rollout.encode(self.actor, self.normalizer, episodes)
        features = encoded.latent.detach()
        command_pose = self.rollout.command_history(episodes, raw_actions).to(device=device, dtype=features.dtype)
        self._feature_cache = None
        def decode(selected=None):
            selected = slice(None) if selected is None else selected
            return self.actor.decode_latent(features[selected], morphology, motor_mask,
                command_pose=command_pose[selected])
        # Draw the whole schedule up front: actor early stopping must not change
        # the critic's budget or the RNG used by the next paired update.
        permutations = [self.rng.permutation(len(episodes)) for _ in range(ppo.update_epochs)]
        with torch.no_grad():
            old_mean = decode().motor_increment_mean.clone()
            old_std = self.actor.motor_noise.stddev.detach().clone()
        metrics, gradient_metrics, stop, optimizer_steps, stopping_kl = [], {}, False, 0, None
        analytic_kls = []
        learning_rates = []
        initial_learning_rate = self.actor_optimizer.param_groups[0]["lr"]
        for epoch, permutation in enumerate(permutations):
            for start in range(0, len(episodes), ppo.minibatch_episodes):
                selected = permutation[start:start+ppo.minibatch_episodes].tolist()
                # Safe only while the complete Meta network is frozen. Cache
                # belongs to this batch; EMG controls are applied before encode.
                command = decode(selected)
                mean = command.motor_increment_mean
                policy = MaskedMotorNormal(mean, motor_mask, std=self.actor.motor_noise.stddev)
                new_log_prob = policy.log_prob(actions[selected])
                if epoch == 0 and start == 0 and (new_log_prob.detach()-old_log_prob[selected]).abs().max() > .005:
                    raise RuntimeError("Unchanged policy did not reproduce stored probabilities; check context/dropout")
                policy_loss, kl, clip_fraction = clipped_policy_loss(new_log_prob, old_log_prob[selected],
                                                advantages[selected], clip_ratio=ppo.clip_ratio)
                if ppo.kl_control == "early_stop" and kl.item() > ppo.target_kl:
                    stopping_kl = float(kl)
                    stop = True
                    break
                entropy = policy.entropy().mean()
                actor_loss = policy_loss - ppo.entropy_coefficient * entropy
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                for name in ("encoder", "command_head", "synergy", "motor_noise"):
                    norm = checked_grad_norm(getattr(self.actor, name).parameters(), name=name)
                    gradient_metrics[name] = max(gradient_metrics.get(name, 0.), norm)
                nn.utils.clip_grad_norm_(self.actor.parameters(), ppo.max_grad_norm)
                learning_rates.append(self.actor_optimizer.param_groups[0]["lr"])
                self.actor_optimizer.step()
                with torch.no_grad():
                    new_mean = decode().motor_increment_mean
                    analytic_kls.append(float(diagonal_gaussian_kl(old_mean, old_std, new_mean,
                                                                 self.actor.motor_noise.stddev, motor_mask)))
                if ppo.kl_control == "adaptive_lr":
                    # Use the exact full-rollout KL after each step on FIXED
                    # recorded command histories. Apply the rate to the NEXT
                    # step, carrying it across updates/checkpoints via Adam.
                    # Unlike rl_games minibatch estimates, this avoids sampled
                    # action/minibatch noise; KL is always relative to rollout.
                    for group in self.actor_optimizer.param_groups:
                        group["lr"] = adaptive_kl_learning_rate(group["lr"], analytic_kls[-1], ppo)
                optimizer_steps += 1
                metrics.append([float(policy_loss.detach()), float(kl),
                                float(clip_fraction), float(entropy.detach())])
            if stop:
                break
        if not metrics:
            raise RuntimeError("PPO made no update; unchanged-policy likelihoods or KL are inconsistent")
        # The value function has its own optimization loop and always sees the
        # same number of minibatches, even when the actor hits the KL guard.
        value_losses = []
        for permutation in permutations:
            for start in range(0, len(episodes), ppo.minibatch_episodes):
                selected = permutation[start:start+ppo.minibatch_episodes].tolist()
                self.critic_optimizer.zero_grad(set_to_none=True)
                value_predictions = self.critic(observations[selected])
                value_loss = (value_predictions-returns[selected]).square().mean()
                (ppo.value_coefficient * value_loss).backward()
                gradient_metrics["critic"] = max(gradient_metrics.get("critic", 0.),
                                                 checked_grad_norm(self.critic.parameters(), name="critic"))
                nn.utils.clip_grad_norm_(self.critic.parameters(), ppo.max_grad_norm)
                self.critic_optimizer.step()
                value_losses.append(float(value_loss.detach()))
        with torch.no_grad():
            final_values = self.critic(observations)
            return_var = returns.var(unbiased=False).clamp_min(1e-12)
            explained_before = 1 - (returns-values).var(unbiased=False) / return_var
            explained_after = 1 - (returns-final_values).var(unbiased=False) / return_var
        average = np.mean(metrics, axis=0)
        return {"policy_loss": float(average[0]), "value_loss": float(np.mean(value_losses)),
                "approx_kl": float(average[1]), "policy_clip_fraction": float(average[2]),
                "entropy": float(average[3]), "kl_early_stop": stop, "stopping_kl": stopping_kl,
                "optimizer_steps": optimizer_steps,
                "kl_control": ppo.kl_control,
                "actor_learning_rate_before": initial_learning_rate,
                "actor_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
                "actor_learning_rates_used": learning_rates,
                "critic_optimizer_steps": len(value_losses),
                "full_batch_analytic_kl": analytic_kls[-1], "peak_analytic_kl": max(analytic_kls),
                "analytic_kl_after_each_step": analytic_kls,
                "critic_explained_variance_before": float(explained_before),
                "critic_explained_variance_after": float(explained_after),
                "gradient_norms": gradient_metrics, "mean_return": float(rewards.sum(-1).mean()),
                "train_motion_cost": float(torch.stack([item.costs for item in outcomes]).mean()),
                "train_geometric_cost": float(np.mean([item.trace["geometric_costs"] for item in outcomes])),
                "train_regularization_cost": float(np.mean([item.trace["regularization_costs"] for item in outcomes])),
                "action_clipped_fraction": float(np.mean([item.clipped_fraction for item in outcomes]))}
