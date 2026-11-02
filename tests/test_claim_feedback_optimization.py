"""CPU checks for signed updates, safeguards, and fair fixed-loss controls."""

import math
import unittest

import torch

from mvpg_run.claim_feedback.optimization import masked_objectives, response_step


def unit(label, indices, weight=1.0):
    return {"claim_id": "claim", "label": label, "weight": weight,
            "token_indices": indices}


class MaskedObjectivesTests(unittest.TestCase):
    def test_claim_means_then_separate_weighted_polarities(self):
        nll = torch.tensor([1., 3., 8., 10., 14.], dtype=torch.float64, requires_grad=True)
        positive, negative, stats = masked_objectives(
            nll, [unit("+", [0, 1]), unit("+", [2], 3), unit("-", [3, 4], .2)])
        self.assertAlmostEqual(positive.item(), 6.5)
        self.assertAlmostEqual(negative.item(), 12)
        self.assertEqual(stats["positive_units"], 2)
        self.assertEqual(stats["negative_tokens"], 2)
        pos_grad = torch.autograd.grad(positive, nll, retain_graph=True)[0]
        neg_grad = torch.autograd.grad(negative, nll)[0]
        torch.testing.assert_close(pos_grad, torch.tensor([.125, .125, .75, 0., 0.], dtype=nll.dtype))
        torch.testing.assert_close(neg_grad, torch.tensor([0., 0., 0., .5, .5], dtype=nll.dtype))

    def test_empty_unknown_and_zero_weight_do_not_create_a_loss(self):
        nll = torch.ones(3, requires_grad=True)
        positive, negative, stats = masked_objectives(
            nll, [unit("+", []), unit("-", [1], 0), unit("?", [2])])
        self.assertIsNone(positive)
        self.assertIsNone(negative)
        self.assertEqual(stats["skipped_units"], 3)

    def test_duplicate_ownership_is_rejected(self):
        for units in ([unit("+", [0, 0])], [unit("+", [0]), unit("-", [0])],
                      [unit("+", [1]), unit("+", [1])]):
            with self.subTest(units=units), self.assertRaises(ValueError):
                masked_objectives(torch.ones(3), units)

    def test_invalid_data_is_not_silently_optimized(self):
        cases = [unit("-", [-1]), unit("+", [3]), unit("+", [.5]),
                 unit("bad", [0]), unit("+", [0], -1), unit("+", [0], float("nan"))]
        for example in cases:
            with self.subTest(example=example), self.assertRaises(ValueError):
                masked_objectives(torch.ones(3), [example])
        with self.assertRaises(ValueError):
            masked_objectives(torch.tensor([float("nan")]), [])
        with self.assertRaises(ValueError):
            masked_objectives(torch.ones(2, 2), [])


class ResponseStepTests(unittest.TestCase):
    def test_partial_conflict_corrects_error_and_preserves_selected_positive(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        positive = 5 + p[0]
        negative = 5 + p[0] + 2 * p[1]
        stats = response_step([p], positive, negative, lr=.1, beta=.25)
        self.assertFalse(stats["skipped"])
        self.assertTrue(stats["conflict"])
        self.assertTrue(stats["projection_applied"])
        torch.testing.assert_close(p, torch.tensor([-.025, .2], dtype=p.dtype))
        self.assertGreater(stats["actual_correction_dot"], 0)
        self.assertLess(stats["actual_positive_dot"], 0)
        self.assertAlmostEqual(stats["post_direction_cosine"], 0.)
        self.assertGreater((p[0] + 2 * p[1]).item(), 0)

    def test_active_positive_learning_is_backed_off_when_it_cancels_correction(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        stats = response_step([p], 5 + p[0], 5 + p[0] + .1 * p[1],
                              lr=.1, beta=1, max_backoffs=20)
        self.assertFalse(stats["skipped"])
        self.assertGreater(stats["backoffs"], 0)
        self.assertLess(stats["beta_used"], .01)
        self.assertGreater(stats["actual_correction_dot"], 0)
        self.assertLessEqual(stats["actual_positive_dot"], 0)

    def test_exact_opposition_skips_without_changing_parameters(self):
        p = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
        before = p.detach().clone()
        stats = response_step([p], p.sum(), p.sum(), lr=.1)
        self.assertTrue(stats["skipped"])
        self.assertEqual(stats["reason"], "no_corrective_direction")
        torch.testing.assert_close(p, before)

    def test_projection_only_has_no_active_positive_encouragement(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        stats = response_step([p], p[0], p[0] + p[1], mode="project_only", lr=.1)
        torch.testing.assert_close(p, torch.tensor([0., .1], dtype=p.dtype))
        self.assertFalse(stats["skipped"])
        self.assertEqual(stats["beta_used"], 0)
        self.assertEqual(stats["actual_positive_dot"], 0.)

    def test_fixed_control_is_a_true_weighted_sum_not_a_hidden_guard(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        stats = response_step([p], p[0], p[0] + 2 * p[1],
                              mode="fixed", lr=.1, beta=.25)
        torch.testing.assert_close(p, torch.tensor([.075, .2], dtype=p.dtype))
        self.assertFalse(stats["projection_applied"])
        self.assertEqual(stats["backoffs"], 0)
        self.assertGreater(stats["actual_positive_dot"], 0)
        self.assertFalse(stats["positive_guard_satisfied"])

    def test_actual_delta_norm_is_capped_not_just_gradient_norm(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        stats = response_step([p], None, 3 * p[0] + 4 * p[1],
                              mode="negative", lr=10, max_update_norm=.1)
        torch.testing.assert_close(p, torch.tensor([.06, .08], dtype=p.dtype))
        self.assertLessEqual(p.detach().norm().item(), .1)
        self.assertLessEqual(stats["actual_update_norm"], .1)
        self.assertLess(stats["norm_scale"], 1)

    def test_post_clipping_absolute_tolerance_is_not_falsely_claimed(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        stats = response_step([p], p[0], p[0] + p[1], lr=1,
                              mode="project_only", max_update_norm=.001,
                              correction_tolerance=.1)
        self.assertFalse(stats["skipped"])
        self.assertTrue(stats["correction_guard_satisfied"])
        self.assertTrue(stats["actual_correction_sign_ok"])
        self.assertFalse(stats["actual_correction_above_tolerance"])
        self.assertLess(stats["actual_correction_dot"], .1)

    def test_near_zero_conflicting_protection_skips_conservatively(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        stats = response_step([p], 1e-9 * p[0], p[0] + p[1], lr=.1)
        self.assertTrue(stats["skipped"])
        self.assertEqual(stats["reason"], "near_zero_protection")
        torch.testing.assert_close(p, torch.zeros_like(p))

    def test_single_groups_and_no_feedback(self):
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        stats = response_step([p], p[0], None, lr=.1)
        self.assertEqual(stats["branch"], "positive_only")
        self.assertFalse(stats["skipped"])
        self.assertLess(p[0].item(), 0)
        p2 = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        stats = response_step([p2], None, p2[0], lr=.1)
        self.assertEqual(stats["branch"], "negative_only")
        self.assertFalse(stats["protection_active"])
        self.assertGreater(p2[0].item(), 0)
        before = p.detach().clone()
        stats = response_step([p], None, None, lr=.1)
        self.assertTrue(stats["skipped"])
        torch.testing.assert_close(p, before)

    def test_unused_parameters_and_constant_loss_are_handled(self):
        p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
        unused = torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))
        stats = response_step([p, unused], p.sum(), None, lr=.1)
        self.assertFalse(stats["skipped"])
        self.assertEqual(unused.item(), 3.)
        before = p.detach().clone()
        stats = response_step([p], torch.tensor(1.), None, lr=.1)
        self.assertTrue(stats["skipped"])
        self.assertEqual(stats["reason"], "zero_direction")
        torch.testing.assert_close(p, before)

    def test_bfloat16_rounding_cannot_silently_report_a_successful_update(self):
        p = torch.nn.Parameter(torch.tensor([1000.], dtype=torch.bfloat16))
        stats = response_step([p], None, p.float().sum(), lr=1e-6)
        self.assertTrue(stats["skipped"])
        self.assertEqual(stats["reason"], "zero_actual_update")
        self.assertEqual(stats["actual_update_norm"], 0.)

    def test_nonfinite_gradient_does_not_partially_mutate_other_parameters(self):
        p = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
        q = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        before = [p.detach().clone(), q.detach().clone()]
        # sqrt(0) has a finite loss but an infinite derivative.
        with self.assertRaises(ValueError):
            response_step([p, q], p + q.sqrt(), None, lr=.1)
        torch.testing.assert_close(p, before[0])
        torch.testing.assert_close(q, before[1])

    def test_configuration_and_nonfinite_losses_are_rejected(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        for kwargs in ({"lr": 0}, {"lr": float("nan")}, {"lr": .1, "beta": -1},
                       {"lr": .1, "max_update_norm": 0}, {"lr": .1, "mode": "adam"},
                       {"lr": .1, "beta_backoff": 1}, {"lr": .1, "max_backoffs": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                response_step([p], p, None, **kwargs)
        with self.assertRaises(ValueError):
            response_step([p], torch.tensor(float("inf")), None, lr=.1)


if __name__ == "__main__":
    unittest.main()
